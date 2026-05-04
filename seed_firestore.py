import os
import sys
import sqlite3
import json
from datetime import datetime

os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/api')))

from src.api.firebase_config import db as firestore_db
from firebase_admin import firestore as firestore_admin

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/fraud_demo.db'))

def seed_to_firestore():
    if not os.path.exists(DB_PATH):
        print(f"SQLite DB not found at {DB_PATH}")
        return
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transactions')
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} transactions in SQLite. Seeding to Firestore...")
    
    batch = firestore_db.batch()
    count = 0
    total = 0
    
    for row in rows:
        d = dict(row)
        
        # Parse reasons safely
        raw_reasons = d.get('reasons', '[]')
        if isinstance(raw_reasons, str):
            try:
                reasons = json.loads(raw_reasons)
            except:
                reasons = []
        else:
            reasons = raw_reasons
            
        doc_ref = firestore_db.collection('transactions').document(d['transaction_id'])
        doc_data = {
            "transaction_id": d.get('transaction_id'),
            "user_id": d.get('user_id'),
            "amount": d.get('amount'),
            "category": d.get('category'),
            "risk_score": d.get('risk_score'),
            "decision": d.get('decision'),
            "reasons": reasons,
            "supervised_confidence": d.get('supervised_confidence'),
            "anomaly_index": d.get('anomaly_index'),
            "location_lat": d.get('location_lat'),
            "location_lon": d.get('location_lon'),
            "timestamp": d.get('timestamp')
        }
        
        # Set created_at correctly for chronological ordering
        if d.get('created_at'):
            # Convert string to Firestore ServerTimestamp or actual datetime
            # Firestore handles ISO format strings gracefully
            doc_data['created_at'] = d.get('created_at')
        else:
            doc_data['created_at'] = firestore_admin.SERVER_TIMESTAMP
            
        batch.set(doc_ref, doc_data)
        
        count += 1
        total += 1
        
        # Firestore batch limit is 500
        if count == 400:
            batch.commit()
            print(f"Committed batch of 400. Total migrated: {total}")
            batch = firestore_db.batch()
            count = 0
            
    if count > 0:
        batch.commit()
        print(f"Committed final batch of {count}. Total migrated: {total}")
        
    conn.close()
    print("Seeding completed successfully.")

if __name__ == '__main__':
    seed_to_firestore()
