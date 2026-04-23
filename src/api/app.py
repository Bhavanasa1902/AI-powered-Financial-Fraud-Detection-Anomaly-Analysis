import os
import sys
import json
import traceback
import pandas as pd
import numpy as np
import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add root project path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocessing import FraudDataPreprocessor
from src.features.engineering import featureEngineering
from src.models.supervised import SupervisedFraudModel
from src.models.unsupervised import UnsupervisedAnomalyDetector
from src.models.deep_learning import DeepAnomalyDetector
from src.models.ensemble import EnsembleScoringEngine

app = Flask(__name__)
CORS(app)

# Global variables to hold models and state
preprocessor = None
supervised_model = None
unsupervised_model = None
ae_model = None
engineer = None
ensemble = None
db_df = None
transactions_log = [] # In-memory queue for Analyst Feed
# Initialize all demo users with NYC as home base for the demo
live_user_base = {f"U{i:05d}": (40.7128, -74.0060) for i in range(1, 21)}

# Mock User Database (20 Users)
USERS = {f"U{i:05d}": {"name": f"Customer {i}", "password": "password123", "balance": 10000.0} for i in range(1, 21)}

# Mock Product Database
PRODUCTS = [
    {"id": "p1", "name": "Premium Smartphone", "price": 999.99, "category": "high_risk_electronics", "image": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&q=80"},
    {"id": "p2", "name": "Designer Watch", "price": 1200.00, "category": "retail", "image": "https://images.unsplash.com/photo-1524592093837-8f355ff19b9a?w=400&q=80"},
    {"id": "p3", "name": "Budget Earbuds", "price": 49.99, "category": "retail", "image": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&q=80"},
    {"id": "p4", "name": "Local Grocery Haul", "price": 150.00, "category": "food", "image": "https://images.unsplash.com/photo-1542838132-92c53300491e?w=400&q=80"},
    {"id": "p5", "name": "Subscription Service", "price": 15.99, "category": "online_services", "image": "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=400&q=80"},
    {"id": "p6", "name": "Last Minute Flight", "price": 850.00, "category": "travel", "image": "https://images.unsplash.com/photo-1436491865332-7a61a109c055?w=400&q=80"}
]

def load_models():
    global preprocessor, supervised_model, unsupervised_model, ae_model, engineer, ensemble, db_df
    print("Loading models into memory...")
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models'))
    
    try:
        preprocessor = FraudDataPreprocessor()
        preprocessor.load_preprocessors(f'{models_dir}/')
        
        supervised_model = SupervisedFraudModel()
        supervised_model.load(f'{models_dir}/supervised_rf.pkl')
        
        unsupervised_model = UnsupervisedAnomalyDetector()
        unsupervised_model.load(f'{models_dir}/isolation_forest.pkl')
        
        # Load autoencoder structure dynamically
        # Re-calculating feature count from expected columns (10 explainable features)
        ae_model = DeepAnomalyDetector(input_dim=10) 
        try:
            ae_model.load(f'{models_dir}/autoencoder.pt', input_dim=10)
        except Exception as e:
            print(f"Warning: Autoencoder load issue: {e}. Will attempt fallback.")
            
        engineer = featureEngineering()
        ensemble = EnsembleScoringEngine()
        
        print("Loading historical database...")
        db_df = pd.read_csv(f'{models_dir}/reference_db.csv')
        db_df['timestamp'] = pd.to_datetime(db_df['timestamp'])
        
        seed_initial_data()
        
        print("System ready for real-time inference.")
        
    except Exception as e:
        print(f"Error loading models. Have you run the training pipeline yet? Error: {str(e)}")

def seed_initial_data():
    """Generates ~200 historical transactions across 20 users to populate charts."""
    global transactions_log
    print("Seeding historical analyst data...")
    start_time = datetime.datetime.now() - datetime.timedelta(days=1)
    
    for i in range(1, 21):
        u_id = f"U{i:05d}"
        for j in range(10):
            ts = start_time + datetime.timedelta(minutes=j*15 + i*5)
            # Create a mix of normal and slightly risky tx
            amount = 20.0 + (i * 5) + (j * 2)
            risk = 5 + (j * 2)
            dec = "ALLOW"
            
            transactions_log.append({
                "status": "success",
                "transaction_id": f"SEED_{u_id}_{j}",
                "user_id": u_id,
                "amount": amount,
                "category": "retail",
                "risk_score": int(risk),
                "decision": dec,
                "models": {"supervised_confidence": risk, "anomaly_index": risk},
                "primary_reasons": [],
                "timestamp": ts.strftime("%H:%M:%S")
            })

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/shop')
def shop_page():
    return render_template('shop.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    u_id = data.get('user_id')
    pwd = data.get('password')
    if u_id in USERS and USERS[u_id]['password'] == pwd:
        # On login, ensure they are reset to NYC home base for the demo session
        live_user_base[u_id] = (40.7128, -74.0060)
        return jsonify({"status": "success", "user": {"id": u_id, "name": USERS[u_id]['name']}})
    return jsonify({"status": "error", "message": "Invalid ID or password"}), 401

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify({"status": "success", "products": PRODUCTS})

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    total_tx = len(transactions_log)
    blocked_tx = len([t for t in transactions_log if t['decision'] == 'BLOCK'])
    flagged_tx = len([t for t in transactions_log if t['decision'] == 'FLAG_FOR_REVIEW'])
    
    # Simple aggregation for recent 24h (mocked)
    return jsonify({
        "status": "success",
        "total": total_tx,
        "blocked": blocked_tx,
        "flagged": flagged_tx,
        "uptime": "99.9%"
    })

@app.route('/api/admin/user/<user_id>/trends', methods=['GET'])
def get_user_trends(user_id):
    user_txs = [t for t in transactions_log if t['user_id'] == user_id]
    # In a real app we'd fetch from a DB; here we use the in-memory log
    # Mocking some historical points if user has few transactions
    trends = []
    for tx in user_txs[-10:]:
        trends.append({
            "time": tx['timestamp'],
            "risk_score": tx['risk_score'],
            "amount": tx['amount']
        })
    return jsonify({"status": "success", "trends": trends})

@app.route('/api/feed', methods=['GET'])
def get_feed():
    # Return the latest 50 transactions for the Operation Control Center
    return jsonify({"status": "success", "feed": transactions_log[-50:]})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract fields from incoming real-time request
        user_id = data.get('user_id', 'U00000') 
        merchant_id = data.get('merchant_id', 'M0000')
        merchant_category = data.get('merchant_category', 'retail')
        amount = float(data.get('amount', 0.0))
        location_lat = float(data.get('location_lat', 0.0))
        location_lon = float(data.get('location_lon', 0.0))
        
        timestamp = datetime.datetime.now()
        
        # 1. Determine user's home base. Default to NYC for demo if not set.
        if user_id not in live_user_base:
            live_user_base[user_id] = (40.7128, -74.0060)
        
        user_base_lat, user_base_lon = live_user_base[user_id]
        
        # Fetch historical context for rolling features
        user_history = db_df[db_df['user_id'] == user_id].copy()
            
        # 2. Build single transaction dataframe
        txn_dict = {
            'transaction_id': [f"TXN_LIVE_{int(timestamp.timestamp())}"],
            'user_id': [user_id],
            'merchant_id': [merchant_id],
            'merchant_category': [merchant_category],
            'amount': [amount],
            'timestamp': [timestamp],
            'location_lat': [location_lat],
            'location_lon': [location_lon],
            'user_base_lat': [user_base_lat],
            'user_base_lon': [user_base_lon]
        }
                
        txn_df = pd.DataFrame(txn_dict)
        
        # Combine with historical user data to compute rolling features (velocity, Z-scores)
        combined_df = pd.concat([user_history, txn_df], ignore_index=True)
        # Apply feature engineering to the combined dataframe
        combined_engineered = engineer.fit_transform(combined_df.copy())
        
        # Extract just the latest transaction's computed features
        latest_txn = combined_engineered.iloc[[-1]].copy()
        
        # 3. Apply standard preprocessing (time splitting, scaling, encoding)
        latest_txn = preprocessor.extract_time_features(latest_txn)
        
        engineered_cols = ['tx_count_12h', 'amount_sum_12h', 'time_since_last_tx', 'geo_distance_km', 'spend_deviation_zscore']
        cat_cols = ['merchant_category']
        num_cols = ['amount', 'hour', 'day_of_week', 'is_weekend'] + engineered_cols
        
        # Fill NA, scale, encode
        latest_txn.fillna(0, inplace=True)
        
        # Capture RAW feature values before scaling (for rule-based thresholds)
        raw_geo_km = float(latest_txn['geo_distance_km'].values[0])
        raw_spend_z = float(latest_txn['spend_deviation_zscore'].values[0])
        raw_amount = float(latest_txn['amount'].values[0])
        
        latest_txn = preprocessor.encode_categorical(latest_txn, cat_cols)
        latest_txn = preprocessor.scale_numerical(latest_txn, num_cols, is_train=False)
        
        features = cat_cols + num_cols
        X = latest_txn[features].values
        
        # 4. Predict
        sup_prob = supervised_model.predict_proba(X)[0]
        iso_score = unsupervised_model.predict_anomaly_score(X)[0]
        try:
            ae_score = ae_model.predict_anomaly_score(X)[0]
        except:
            ae_score = 0.0
            
        print(f"DEBUG: user={user_id} amt={amount} sup={sup_prob:.3f} iso={iso_score:.3f} ae={ae_score:.3f}")
        
        # 5. Ensemble Score (base)
        # For the demo, suppress anomaly scores if the transaction is local (< 500km)
        # to avoid baseline noise from the unsupervised detectors.
        adj_iso = iso_score if raw_geo_km > 500 else 0.0
        adj_ae = ae_score if raw_geo_km > 500 else 0.0
        
        base_risk = ensemble.calculate_risk_score(
            np.array([sup_prob]), 
            np.array([adj_iso]), 
            np.array([adj_ae])
        )[0]
        
        # 6. Contextual rule-based boosting (Thresholds tuned for NYC to London distance (~5500km))
        boost = 0
        reasons = []
        
        # Thresholds tuned for NYC to London distance (~5500km)
        if raw_geo_km > 500: 
            reasons.append("Unusual Location")
            boost += 20
        if raw_geo_km > 3000:
            boost += 15
            
        if raw_spend_z > 2.0:
            reasons.append("Unusual Spend Amount")
            boost += 10
        if raw_spend_z > 4.0:
            boost += 10
            
        if raw_amount > 500 and raw_geo_km > 500:
            reasons.append("High-value overseas purchase")
            boost += 15
            
        if sup_prob > 0.6:
            reasons.append("Matches known fraud patterns")
              
        risk_score = min(int(base_risk) + boost, 100)
        
        decision = ensemble.get_decision(risk_score)
        
        if not reasons and decision != "ALLOW":
            reasons.append("System Anomaly Detected")

        result = {
            "status": "success",
            "transaction_id": txn_dict['transaction_id'][0],
            "user_id": user_id,
            "amount": amount,
            "category": merchant_category,
            "risk_score": int(risk_score),
            "decision": decision,
            "models": {
                "supervised_confidence": round(float(sup_prob) * 100, 2),
                "anomaly_index": round(float(iso_score) * 100, 2)
            },
            "primary_reasons": reasons,
            "timestamp": timestamp.strftime("%H:%M:%S")
        }
        
        # Append to live in-memory feed
        transactions_log.append(result)
        # Keep log trimmed
        if len(transactions_log) > 100:
            transactions_log.pop(0)
            
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    load_models()
    # Run on port 5001 to avoid common 5000 conflicts
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
