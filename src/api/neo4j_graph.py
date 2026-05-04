"""
neo4j_graph.py — Neo4j AuraDB integration for fraud ring detection.

Graph model:
  (User {uid, name}) -[:MADE]-> (Transaction {tx_id, amount, risk_score, decision})
  (Transaction)      -[:FROM_IP]-> (IPNode {ip})
  (Transaction)      -[:USES_CARD]-> (CardHash {hash})
  (Transaction)      -[:IN_CATEGORY]-> (Category {name})

Fraud ring queries:
  1. Shared IP      — multiple users using the same originating IP
  2. Shared card    — multiple users sharing a card hash (stolen card syndicate)
  3. High-risk hub  — IPs linked to ≥N blocked transactions
"""

import os
import hashlib
import traceback
from typing import Optional

# ── Driver singleton ──────────────────────────────────────────────────────────

_driver = None

def _get_driver():
    global _driver
    if _driver is not None:
        return _driver

    uri  = os.environ.get("NEO4J_URI", "")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd  = os.environ.get("NEO4J_PASSWORD", "")

    if not uri or not pwd:
        return None   # Neo4j not configured — degrade gracefully

    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(uri, auth=(user, pwd))
        _driver.verify_connectivity()
        print("[Neo4j] Connected to AuraDB successfully.")
    except Exception as e:
        print(f"[Neo4j] Connection failed (non-fatal): {e}")
        _driver = None

    return _driver


def _hash_card(card_number: str) -> str:
    """SHA-256 of the last 4 digits + card length — never stores full PAN."""
    clean = card_number.replace(" ", "").replace("-", "")
    digest = hashlib.sha256(f"{clean[-4:]}:{len(clean)}".encode()).hexdigest()[:16]
    return digest


# ── Write a transaction into the graph ───────────────────────────────────────

def write_transaction_graph(
    user_id: str,
    user_name: str,
    tx_id: str,
    amount: float,
    category: str,
    risk_score: int,
    decision: str,
    ip_address: str,
    card_number: str,
):
    """
    Upserts all nodes/relationships for one transaction.
    Designed to be called asynchronously — any failure is logged but non-fatal.
    """
    driver = _get_driver()
    if not driver:
        return

    card_hash = _hash_card(card_number) if card_number else "UNKNOWN"

    cypher = """
    MERGE (u:User {uid: $uid})
      ON CREATE SET u.name = $name
      ON MATCH  SET u.name = $name

    MERGE (tx:Transaction {tx_id: $tx_id})
      ON CREATE SET tx.amount     = $amount,
                    tx.risk_score = $risk_score,
                    tx.decision   = $decision,
                    tx.category   = $category

    MERGE (ip:IPNode {ip: $ip})
    MERGE (ch:CardHash {hash: $card_hash})
    MERGE (cat:Category {name: $category})

    MERGE (u)-[:MADE]->(tx)
    MERGE (tx)-[:FROM_IP]->(ip)
    MERGE (tx)-[:USES_CARD]->(ch)
    MERGE (tx)-[:IN_CATEGORY]->(cat)
    MERGE (u)-[:USES_CARD]->(ch)
    """

    try:
        with driver.session() as session:
            session.run(cypher, {
                "uid": user_id, "name": user_name,
                "tx_id": tx_id, "amount": amount,
                "risk_score": risk_score, "decision": decision,
                "category": category,
                "ip": ip_address or "UNKNOWN",
                "card_hash": card_hash,
            })
    except Exception:
        print(f"[Neo4j] write_transaction_graph error:\n{traceback.format_exc()}")


# ── Fraud ring queries ────────────────────────────────────────────────────────

def query_shared_ip_rings(min_users: int = 2) -> list[dict]:
    """
    Returns IPs used by ≥ min_users distinct users — classic IP-sharing fraud ring.
    """
    driver = _get_driver()
    if not driver:
        return []

    cypher = """
    MATCH (u:User)-[:MADE]->(tx:Transaction)-[:FROM_IP]->(ip:IPNode)
    WITH ip.ip AS ip_addr, collect(DISTINCT u.uid) AS uids,
         collect(DISTINCT tx.tx_id) AS txs,
         sum(CASE WHEN tx.decision = 'BLOCK' THEN 1 ELSE 0 END) AS blocked
    WHERE size(uids) >= $min_users
    RETURN ip_addr, uids, size(txs) AS tx_count, blocked
    ORDER BY size(uids) DESC
    LIMIT 20
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, {"min_users": min_users})
            return [dict(r) for r in result]
    except Exception:
        print(f"[Neo4j] query_shared_ip_rings error:\n{traceback.format_exc()}")
        return []


def query_shared_card_rings(min_users: int = 2) -> list[dict]:
    """
    Returns card hashes used by ≥ min_users distinct users — stolen card syndicate.
    """
    driver = _get_driver()
    if not driver:
        return []

    cypher = """
    MATCH (u:User)-[:USES_CARD]->(ch:CardHash)
    WITH ch.hash AS card_hash, collect(DISTINCT u.uid) AS uids,
         collect(DISTINCT u.name) AS names
    WHERE size(uids) >= $min_users
    RETURN card_hash, uids, names, size(uids) AS user_count
    ORDER BY size(uids) DESC
    LIMIT 20
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, {"min_users": min_users})
            return [dict(r) for r in result]
    except Exception:
        print(f"[Neo4j] query_shared_card_rings error:\n{traceback.format_exc()}")
        return []


def query_high_risk_ips(min_blocks: int = 2) -> list[dict]:
    """
    Returns IPs associated with ≥ min_blocks BLOCK decisions — bad actor IPs.
    """
    driver = _get_driver()
    if not driver:
        return []

    cypher = """
    MATCH (tx:Transaction)-[:FROM_IP]->(ip:IPNode)
    WHERE tx.decision = 'BLOCK'
    WITH ip.ip AS ip_addr, count(tx) AS block_count,
         collect(DISTINCT tx.tx_id)[0..5] AS sample_txs
    WHERE block_count >= $min_blocks
    RETURN ip_addr, block_count, sample_txs
    ORDER BY block_count DESC
    LIMIT 20
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, {"min_blocks": min_blocks})
            return [dict(r) for r in result]
    except Exception:
        print(f"[Neo4j] query_high_risk_ips error:\n{traceback.format_exc()}")
        return []


def get_neo4j_status() -> dict:
    """Returns connection status and basic graph stats for the dashboard status panel."""
    driver = _get_driver()
    if not driver:
        return {"connected": False, "nodes": 0, "relationships": 0}

    try:
        with driver.session() as session:
            n = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            return {"connected": True, "nodes": n, "relationships": r}
    except Exception:
        return {"connected": False, "nodes": 0, "relationships": 0}
