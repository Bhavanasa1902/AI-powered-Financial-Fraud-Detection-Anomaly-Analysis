import os
import sys
import json
import time
import traceback
import threading

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add root project path AND api directory to import src + firebase_config modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from firebase_config import db as firestore_db, verify_firebase_token
import neo4j_graph

app = Flask(__name__)
CORS(app)

# ─── Auth Decorator ───────────────────────────────────────────────────────────

def require_auth(f):
    """Decorator to protect routes: verifies Firebase ID token from Authorization header."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        decoded = verify_firebase_token(request)
        if not decoded:
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        # Check admin role in Firestore (uses TTL cache to avoid a read per request)
        uid = decoded['uid']
        role = _get_user_role(uid)
        if role != 'admin':
            return jsonify({"status": "error", "message": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

# ─── TTL User Cache (avoids repeated Firestore reads per request) ─────────────
#
# Stores { uid: {"display_name": str, "role": str, "expires": float} }
# Default TTL: 5 minutes — users don't change often.

_USER_CACHE: dict = {}
_USER_CACHE_TTL = 300  # seconds

def _get_user_doc(uid: str) -> dict:
    """Fetch user document from Firestore, with a 5-min TTL cache."""
    now = time.monotonic()
    entry = _USER_CACHE.get(uid)
    if entry and entry["expires"] > now:
        return entry

    # Cache miss — one Firestore read
    try:
        doc = firestore_db.collection('users').document(uid).get()
        data = doc.to_dict() if doc.exists else {}
    except Exception:
        data = {}

    cached = {
        "display_name": data.get("display_name", uid),
        "role": data.get("role", ""),
        "expires": now + _USER_CACHE_TTL
    }
    _USER_CACHE[uid] = cached
    return cached


def _get_user_role(uid: str) -> str:
    return _get_user_doc(uid).get("role", "")


def _get_user_display_name(uid: str) -> str:
    return _get_user_doc(uid).get("display_name", uid)

# ─── Real-time RAM listener (zero extra reads after initial sync) ─────────────

live_transactions: dict = {}

def on_snapshot(col_snapshot, changes, read_time):
    """Syncs every Firestore add/update into RAM. No polling needed."""
    for change in changes:
        doc = change.document
        live_transactions[doc.id] = doc.to_dict()

try:
    firestore_db.collection('transactions').on_snapshot(on_snapshot)
except Exception as e:
    print(f"Warning: Failed to attach Firestore snapshot listener: {e}")

# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def bank_login_page():
    return render_template('bank_login.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('admin.html')

# ─── API: Stats ───────────────────────────────────────────────────────────────

@app.route('/api/bank/stats', methods=['GET'])
@require_auth
def get_bank_stats():
    try:
        total_tx = len(live_transactions)
        blocked_tx = 0
        flagged_tx = 0

        for doc in live_transactions.values():
            d = doc.get('decision')
            if d == 'BLOCK':
                blocked_tx += 1
            elif d in ['FLAG_FOR_REVIEW', 'STEP_UP_AUTH']:
                flagged_tx += 1

        return jsonify({
            "status": "success",
            "total": total_tx,
            "blocked": blocked_tx,
            "flagged": flagged_tx,
            "uptime": "99.9%"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── API: User Trend ──────────────────────────────────────────────────────────

@app.route('/api/bank/user/<user_id>/trends', methods=['GET'])
@require_auth
def get_bank_user_trends(user_id):
    rows = [r for r in live_transactions.values() if r.get('user_id') == user_id]
    rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
    rows = rows[:10]

    trends = []
    for r in rows:
        trends.insert(0, {
            "time": r.get('timestamp', ''),
            "risk_score": r.get('risk_score', 0),
            "amount": r.get('amount', 0)
        })

    return jsonify({"status": "success", "trends": trends})

# ─── API: Feed ────────────────────────────────────────────────────────────────

@app.route('/api/bank/feed', methods=['GET'])
@require_auth
def get_bank_feed():
    limit = int(request.args.get('limit', 150))
    offset = int(request.args.get('offset', 0))
    decision = request.args.get('decision', 'ALL')

    rows = list(live_transactions.values())
    if decision != 'ALL':
        if decision == 'FLAG_FOR_REVIEW':
            rows = [r for r in rows if r.get('decision') in ['FLAG_FOR_REVIEW', 'STEP_UP_AUTH']]
        else:
            rows = [r for r in rows if r.get('decision') == decision]

    rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
    rows = rows[offset:offset + limit]

    feed = []
    for r in reversed(rows):
        uid = r.get('user_id', '')
        name = _get_user_display_name(uid) if uid else uid

        raw_reasons = r.get('reasons', [])
        if isinstance(raw_reasons, str):
            try:
                primary_reasons = json.loads(raw_reasons)
            except Exception:
                primary_reasons = []
        else:
            primary_reasons = raw_reasons

        feed.append({
            "status": "success",
            "transaction_id": r.get('transaction_id', ''),
            "user_id": uid,
            "user_name": name,
            "amount": r.get('amount', 0),
            "category": r.get('category', ''),
            "risk_score": r.get('risk_score', 0),
            "decision": r.get('decision', 'ALLOW'),
            "models": {
                "supervised_confidence": r.get('supervised_confidence', 0),
                "anomaly_index": r.get('anomaly_index', 0)
            },
            "primary_reasons": primary_reasons,
            "timestamp": r.get('timestamp', '')
        })

    return jsonify({"status": "success", "feed": feed})

# ─── Retrain Logic (shared by scheduler + manual endpoint) ───────────────────

_retrain_lock = threading.Lock()

# State visible to the dashboard via /api/bank/retrain/status
_retrain_state = {
    "running": False,
    "last_run": None,       # ISO timestamp string
    "last_result": None,    # "success" | "error"
    "last_message": "",
    "next_run": None,       # ISO timestamp string (set by scheduler)
    "new_rows": 0,
}

_LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../retrain.log'))


def _do_retrain_core():
    """
    Retrain all models using live_transactions RAM store + reference_db.csv.
    No extra Firestore stream — we reuse the already-synced in-memory data.
    Returns (success: bool, message: str, new_rows: int).
    """
    import pandas as pd
    import numpy as np

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models'))
    ref_path = os.path.join(models_dir, 'reference_db.csv')

    try:
        ref_df = pd.read_csv(ref_path)
        ref_df['timestamp'] = pd.to_datetime(ref_df['timestamp'], errors='coerce')
    except Exception as e:
        return False, f"Failed to load reference_db.csv: {e}", 0

    # Pull non-seed transactions straight from RAM — zero Firestore reads
    new_data = []
    for d in live_transactions.values():
        tid = str(d.get('transaction_id', ''))
        if tid.startswith('SEED_'):
            continue
        new_data.append({
            'transaction_id': tid,
            'user_id': d.get('user_id'),
            'amount': d.get('amount'),
            'merchant_category': d.get('category'),
            'location_lat': d.get('location_lat'),
            'location_lon': d.get('location_lon'),
            'timestamp': d.get('timestamp'),
            'decision': d.get('decision'),
            'is_fraud': 1 if d.get('decision') == 'BLOCK' else 0,
        })

    if not new_data:
        return True, "No new transactions to retrain on. Models unchanged.", 0

    new_df = pd.DataFrame(new_data)
    new_count = len(new_df)

    try:
        from src.data.preprocessing import FraudDataPreprocessor
        from src.features.engineering import featureEngineering
        from src.models.supervised import SupervisedFraudModel
        from src.models.unsupervised import UnsupervisedAnomalyDetector
        from src.models.deep_learning import DeepAnomalyDetector

        print(f"\n=== AUTO-RETRAIN: Incorporating {new_count} new transactions ===")

        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
        new_df['transaction_id'] = [f"RETRAIN_{i}" for i in range(len(new_df))]
        new_df['merchant_id'] = 'M0000'
        new_df['user_base_lat'] = 40.7128
        new_df['user_base_lon'] = -74.0060

        combined = pd.concat([ref_df, new_df], ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce').fillna(pd.Timestamp.now())

        eng = featureEngineering()
        combined = eng.fit_transform(combined)

        prep = FraudDataPreprocessor()
        combined = prep.extract_time_features(combined)

        cat_cols = ['merchant_category']
        engineered_cols = ['tx_count_12h', 'amount_sum_12h', 'time_since_last_tx', 'geo_distance_km', 'spend_deviation_zscore']
        num_cols = ['amount', 'hour', 'day_of_week', 'is_weekend'] + engineered_cols

        combined.fillna(0, inplace=True)
        combined = prep.encode_categorical(combined, cat_cols)
        combined = prep.scale_numerical(combined, num_cols, is_train=True)

        features = cat_cols + num_cols
        X = combined[features].values
        y = combined['is_fraud'].values if 'is_fraud' in combined.columns else np.zeros(len(combined))

        sup = SupervisedFraudModel(model_type='random_forest')
        sup.build_model()
        sup.train(X, y)
        sup.save(os.path.join(models_dir, 'supervised_rf.pkl'))

        normal_X = X[y == 0]
        unsup = UnsupervisedAnomalyDetector()
        unsup.train(normal_X)
        unsup.save(os.path.join(models_dir, 'isolation_forest.pkl'))

        ae = DeepAnomalyDetector(input_dim=X.shape[1])
        ae.train(normal_X)
        ae.save(os.path.join(models_dir, 'autoencoder.pt'))

        prep.save_preprocessors(os.path.join(models_dir, ''))

        print("=== AUTO-RETRAIN COMPLETE ===\n")
        return True, f"Retrained on {new_count} new transactions successfully.", new_count

    except Exception:
        msg = traceback.format_exc()
        print(f"Retrain error:\n{msg}")
        return False, f"Retrain failed: {msg[:200]}", new_count


def _run_retrain_background(triggered_by: str = "manual"):
    """Thread-safe wrapper: prevents concurrent retrains, updates _retrain_state."""
    if not _retrain_lock.acquire(blocking=False):
        print(f"[Retrain] Skipped ({triggered_by}) — another retrain already running.")
        return

    from datetime import datetime, timezone
    _retrain_state["running"] = True
    _retrain_state["last_run"] = datetime.now(timezone.utc).isoformat()
    _retrain_state["last_result"] = None
    _retrain_state["last_message"] = "Running…"

    try:
        success, message, new_rows = _do_retrain_core()
        _retrain_state["last_result"] = "success" if success else "error"
        _retrain_state["last_message"] = message
        _retrain_state["new_rows"] = new_rows
        result_label = "SUCCESS" if success else "ERROR"
        print(f"[Retrain] Done ({triggered_by}): {message}")

        # Append structured record to retrain.log
        try:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            with open(_LOG_PATH, 'a', encoding='utf-8') as lf:
                lf.write(
                    f"\n[{ts}] trigger={triggered_by} result={result_label} "
                    f"new_rows={new_rows} | {message}\n"
                )
        except Exception as log_err:
            print(f"[Retrain] Warning: could not write retrain.log: {log_err}")
    finally:
        _retrain_state["running"] = False
        _retrain_lock.release()

# ─── Automated Scheduler ──────────────────────────────────────────────────────

RETRAIN_INTERVAL_HOURS = 6  # retrain every 6 hours automatically

def _schedule_retrain():
    """Fires _run_retrain_background on a fixed interval, forever."""
    from datetime import datetime, timezone, timedelta

    while True:
        # Sleep first so the first run is after RETRAIN_INTERVAL_HOURS, not at startup
        next_run_dt = datetime.now(timezone.utc) + timedelta(hours=RETRAIN_INTERVAL_HOURS)
        _retrain_state["next_run"] = next_run_dt.isoformat()
        print(f"[Scheduler] Next auto-retrain at {next_run_dt.strftime('%Y-%m-%d %H:%M UTC')}")

        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)

        print("[Scheduler] Triggering scheduled retrain…")
        _run_retrain_background(triggered_by="scheduler")

# Start the scheduler daemon thread once
_scheduler_thread = threading.Thread(target=_schedule_retrain, daemon=True, name="AutoRetrainScheduler")
_scheduler_thread.start()

# ─── API: Retrain (manual trigger + status) ───────────────────────────────────

@app.route('/api/bank/retrain', methods=['POST'])
@require_auth
def retrain_models():
    """Manual retrain trigger — spawns background thread immediately."""
    if _retrain_state["running"]:
        return jsonify({
            "status": "error",
            "message": "A retrain is already in progress. Please wait."
        }), 409

    thread = threading.Thread(
        target=_run_retrain_background,
        kwargs={"triggered_by": "manual"},
        daemon=True
    )
    thread.start()

    return jsonify({
        "status": "success",
        "message": "Manual retrain started. Models will update in the background.",
    })


@app.route('/api/bank/retrain/status', methods=['GET'])
@require_auth
def retrain_status():
    """Returns the current retrain state so the dashboard can poll it."""
    return jsonify({
        "status": "success",
        "retrain": {
            "running": _retrain_state["running"],
            "last_run": _retrain_state["last_run"],
            "last_result": _retrain_state["last_result"],
            "last_message": _retrain_state["last_message"],
            "next_run": _retrain_state["next_run"],
            "new_rows": _retrain_state["new_rows"],
            "interval_hours": RETRAIN_INTERVAL_HOURS,
        }
    })


# ─── Firestore RAM-based Fraud Ring Analysis (always-on fallback) ────────────
#
# When Neo4j is not configured, we mine fraud rings directly from the
# live_transactions RAM store — no extra reads, no external dependency.
#
# Detection logic:
#   Shared-location rings  → transactions from same rounded lat/lon cluster,
#                            different user_ids  (≈ shared IP / same device)
#   Risky user clusters    → users with 2+ BLOCK decisions
#   High-value anomalies   → high-risk score + BLOCK, grouped by category

def _ram_shared_location_rings(min_users: int = 2) -> list:
    """
    Group transactions by rounded lat/lon (1 decimal ≈ 11km radius).
    Clusters with users_count >= min_users are reported as shared-location rings.
    """
    from collections import defaultdict
    clusters: dict = defaultdict(lambda: {"uids": set(), "tx_ids": [], "blocked": 0})

    for doc in live_transactions.values():
        lat = doc.get("location_lat")
        lon = doc.get("location_lon")
        uid = doc.get("user_id", "")
        tid = doc.get("transaction_id", "")
        decision = doc.get("decision", "ALLOW")

        if lat is None or lon is None or not uid:
            continue

        # Round to 1 decimal place ≈ ~11 km accuracy
        key = f"{round(float(lat), 1)},{round(float(lon), 1)}"
        clusters[key]["uids"].add(uid)
        clusters[key]["tx_ids"].append(tid)
        if decision == "BLOCK":
            clusters[key]["blocked"] += 1

    rings = []
    for loc_key, data in clusters.items():
        if len(data["uids"]) >= min_users:
            lat_str, lon_str = loc_key.split(",")
            rings.append({
                "ip_addr": f"Location {lat_str}°,{lon_str}° (cluster)",
                "uids": sorted(data["uids"]),
                "tx_count": len(data["tx_ids"]),
                "blocked": data["blocked"],
            })

    rings.sort(key=lambda x: len(x["uids"]), reverse=True)
    return rings[:10]


def _ram_high_risk_users(min_blocks: int = 2) -> list:
    """
    Users with min_blocks or more BLOCK decisions — high-risk account clusters.
    Displayed in the 'Stolen Card Syndicates' panel as high-risk account groups.
    """
    from collections import defaultdict
    user_stats: dict = defaultdict(lambda: {"blocks": 0, "tx_ids": [], "categories": set(), "names": set()})

    for doc in live_transactions.values():
        uid  = doc.get("user_id", "")
        name = doc.get("user_name") or uid
        decision = doc.get("decision", "ALLOW")
        cat  = doc.get("category", "")
        tid  = doc.get("transaction_id", "")

        if not uid:
            continue

        user_stats[uid]["names"].add(name)
        user_stats[uid]["categories"].add(cat)
        user_stats[uid]["tx_ids"].append(tid)
        if decision == "BLOCK":
            user_stats[uid]["blocks"] += 1

    # Also resolve display names from the TTL user cache
    result = []
    for uid, stats in user_stats.items():
        if stats["blocks"] >= min_blocks:
            cached_name = _get_user_display_name(uid)
            all_names = stats["names"] | {cached_name}
            result.append({
                "card_hash": f"USER-{uid[:8]}",
                "uids": [uid],
                "names": sorted(all_names),
                "user_count": 1,
                "block_count": stats["blocks"],
                "categories": sorted(stats["categories"]),
            })

    result.sort(key=lambda x: x["block_count"], reverse=True)
    return result[:10]


def _ram_high_risk_locations(min_blocks: int = 2) -> list:
    """
    Location clusters with multiple BLOCK decisions — high-risk origin zones.
    Displayed in the 'High-Risk IP Addresses' panel.
    """
    from collections import defaultdict
    loc_blocks: dict = defaultdict(lambda: {"blocks": 0, "tx_ids": []})

    for doc in live_transactions.values():
        lat = doc.get("location_lat")
        lon = doc.get("location_lon")
        decision = doc.get("decision", "ALLOW")
        tid = doc.get("transaction_id", "")

        if lat is None or lon is None:
            continue
        if decision != "BLOCK":
            continue

        key = f"{round(float(lat), 1)},{round(float(lon), 1)}"
        loc_blocks[key]["blocks"] += 1
        loc_blocks[key]["tx_ids"].append(tid)

    result = []
    for loc_key, data in loc_blocks.items():
        if data["blocks"] >= min_blocks:
            lat_str, lon_str = loc_key.split(",")
            result.append({
                "ip_addr": f"Zone {lat_str}°,{lon_str}°",
                "block_count": data["blocks"],
                "sample_txs": data["tx_ids"][:5],
            })

    result.sort(key=lambda x: x["block_count"], reverse=True)
    return result[:10]


# ─── Neo4j Fraud Ring Detection Endpoints ────────────────────────────────────

@app.route('/api/bank/fraud-rings', methods=['GET'])
@require_auth
def get_fraud_rings():
    """
    Dual-mode fraud ring detection:
    - When Neo4j is connected → uses full graph queries (shared IPs, card hashes)
    - When Neo4j is offline   → mines live_transactions RAM store (always available)
    """
    try:
        neo4j_ok = neo4j_graph.get_neo4j_status().get("connected", False)

        if neo4j_ok:
            # ── Neo4j path: precise graph-based detection ──────────────────
            raw_ips   = neo4j_graph.query_shared_ip_rings(min_users=2)
            raw_cards = neo4j_graph.query_shared_card_rings(min_users=2)
            raw_badip = neo4j_graph.query_high_risk_ips(min_blocks=2)

            def _clean(rows):
                out = []
                for r in rows:
                    clean = {}
                    for k, v in r.items():
                        clean[k] = list(v) if (hasattr(v, '__iter__') and not isinstance(v, str)) else v
                    out.append(clean)
                return out

            shared_ips   = _clean(raw_ips)
            shared_cards = _clean(raw_cards)
            bad_ips      = _clean(raw_badip)
            source       = "neo4j"

        else:
            # ── RAM fallback: mine live_transactions dict directly ──────────
            shared_ips   = _ram_shared_location_rings(min_users=2)
            shared_cards = _ram_high_risk_users(min_blocks=2)
            bad_ips      = _ram_high_risk_locations(min_blocks=2)
            source       = "firestore_ram"

        return jsonify({
            "status": "success",
            "source": source,
            "shared_ip_rings":  shared_ips,
            "shared_card_rings": shared_cards,
            "high_risk_ips":    bad_ips,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/bank/graph-status', methods=['GET'])
@require_auth
def get_graph_status():
    """Returns Neo4j connectivity + graph size, or RAM fallback stats."""
    status = neo4j_graph.get_neo4j_status()
    if not status["connected"]:
        # Report RAM fallback stats so dashboard always shows something
        status["nodes"]         = len(live_transactions)
        status["relationships"] = sum(
            1 for d in live_transactions.values() if d.get("decision") == "BLOCK"
        )
        status["source"] = "firestore_ram"
    else:
        status["source"] = "neo4j"
    return jsonify({"status": "success", "graph": status})


if __name__ == '__main__':
    print("Starting Bank Dashboard on port 5002…")
    print(f"Auto-retrain scheduler active — every {RETRAIN_INTERVAL_HOURS}h.")
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
