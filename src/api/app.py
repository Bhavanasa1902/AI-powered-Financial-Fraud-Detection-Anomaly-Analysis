import os
import sys
import json
import sqlite3
import traceback
import threading
import pandas as pd
import numpy as np
import datetime
import shap
from dotenv import load_dotenv
import os
load_dotenv()

from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Add root project path AND api directory to import src + firebase_config modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from firebase_config import db as firestore_db, verify_firebase_token
from firebase_admin import auth as firebase_auth

from src.data.preprocessing import FraudDataPreprocessor
from src.features.engineering import featureEngineering
from src.models.supervised import SupervisedFraudModel
from src.models.unsupervised import UnsupervisedAnomalyDetector
from src.models.deep_learning import DeepAnomalyDetector
from src.models.ensemble import EnsembleScoringEngine
from src.api.notifications import send_transaction_notification
import neo4j_graph

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Global variables to hold models and state
preprocessor = None
supervised_model = None
unsupervised_model = None
ae_model = None
engineer = None
ensemble = None
db_df = None
# Initialize all demo users with NYC as home base for the demo
live_user_base = {f"U{i:05d}": (40.7128, -74.0060) for i in range(1, 21)}

# ─── Mock User Database (20 Users) with Per-User Payment Credentials ───
FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Casey", "Taylor",
    "Riley", "Quinn", "Avery", "Blake", "Cameron",
    "Drew", "Emery", "Finley", "Harper", "Jamie",
    "Kendall", "Lane", "Micah", "Noel", "Parker"
]
LAST_NAMES = [
    "Johnson", "Chen", "Williams", "Patel", "Garcia",
    "Kim", "Martinez", "Anderson", "Thompson", "Nakamura",
    "Davis", "Wilson", "Moore", "Taylor", "White",
    "Harris", "Clark", "Lewis", "Robinson", "Walker"
]
CARD_PREFIXES = ["4532", "4716", "5425", "5168", "4929", "4539", "5307", "4485", "5218", "4024",
                 "4556", "5294", "4916", "5102", "4738", "5387", "4652", "5043", "4831", "5276"]
STREETS = [
    "381 Park Ave", "742 Elm St", "55 Broadway", "128 Oak Blvd", "900 Pine Rd",
    "217 Maple Dr", "463 Cedar Ln", "88 Birch Way", "1550 Walnut St", "320 Spruce Ave",
    "611 Main St", "77 River Rd", "203 Lake Dr", "445 Hill Ct", "160 Valley Rd",
    "892 Ocean Blvd", "34 Summit Ave", "510 Forest Ln", "275 Bridge St", "1020 Harbor Dr"
]
CITIES = [
    "New York, NY", "San Francisco, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "Austin, TX",
    "Jacksonville, FL", "Columbus, OH", "Charlotte, NC", "Indianapolis, IN", "Seattle, WA",
    "Denver, CO", "Boston, MA", "Nashville, TN", "Portland, OR", "Las Vegas, NV"
]

USERS = {}
for i in range(1, 21):
    uid = f"U{i:05d}"
    idx = i - 1
    card_mid = f"{1000 + i*111:04d} {2000 + i*73:04d} {3000 + i*37:04d}"
    USERS[uid] = {
        "name": f"{FIRST_NAMES[idx]} {LAST_NAMES[idx]}",
        "password": "password123",
        "balance": round(5000.0 + i * 500, 2),
        "card_name": f"{FIRST_NAMES[idx]} {LAST_NAMES[idx]}",
        "card_number": f"{CARD_PREFIXES[idx]} {card_mid}",
        "card_expiry": f"{((i * 3) % 12) + 1:02d}/{26 + (i % 4)}",
        "card_cvv": f"{100 + i * 37:03d}",
        "billing_address": f"{STREETS[idx]}, {CITIES[idx]}"
    }

# Mock Product Database — all images use verified Unsplash photo IDs
PRODUCTS = [
    # Electronics
    {"id": "p1", "name": "Premium Smartphone", "price": 999.99, "category": "high_risk_electronics", "image": "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?auto=format&fit=crop&w=400&q=80"},
    {"id": "p2", "name": "Pro Gaming Laptop", "price": 1899.00, "category": "high_risk_electronics", "image": "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?auto=format&fit=crop&w=400&q=80"},
    {"id": "p3", "name": "Noise Cancelling Headphones", "price": 349.99, "category": "high_risk_electronics", "image": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=400&q=80"},
    {"id": "p4", "name": "Mirrorless 4K Camera", "price": 1299.00, "category": "high_risk_electronics", "image": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=400&q=80"},
    
    # Retail / Fashion
    {"id": "p5", "name": "Classic Designer Watch", "price": 450.00, "category": "retail", "image": "https://images.unsplash.com/photo-1523170335258-f5ed11844a49?auto=format&fit=crop&w=400&q=80"},
    {"id": "p6", "name": "Premium Leather Handbag", "price": 320.00, "category": "retail", "image": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?auto=format&fit=crop&w=400&q=80"},
    {"id": "p7", "name": "Running Performance Sneakers", "price": 130.00, "category": "retail", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=400&q=80"},
    {"id": "p8", "name": "Vintage Denim Jacket", "price": 85.00, "category": "retail", "image": "https://images.unsplash.com/photo-1551028719-00167b16eac5?auto=format&fit=crop&w=400&q=80"},
    
    # Travel
    {"id": "p9", "name": "Last Minute Flight (NYC → MIA)", "price": 450.00, "category": "travel", "image": "https://images.unsplash.com/photo-1436491865332-7a61a109c055?auto=format&fit=crop&w=400&q=80"},
    {"id": "p10", "name": "Luxury Oceanfront Resort", "price": 850.00, "category": "travel", "image": "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?auto=format&fit=crop&w=400&q=80"},
    {"id": "p11", "name": "Weekend SUV Rental", "price": 180.00, "category": "travel", "image": "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=400&q=80"},
    {"id": "p12", "name": "VIP Music Festival Pass", "price": 550.00, "category": "travel", "image": "https://images.unsplash.com/photo-1470229722913-7c0e2dbbafd3?auto=format&fit=crop&w=400&q=80"},
    
    # Groceries / Food
    {"id": "p13", "name": "Weekly Grocery Haul", "price": 125.00, "category": "food", "image": "https://images.unsplash.com/photo-1543168256-418811576931?auto=format&fit=crop&w=400&q=80"},
    {"id": "p14", "name": "Artisanal Coffee Beans 3-Pack", "price": 45.00, "category": "food", "image": "https://images.unsplash.com/photo-1447933601403-0c6688de566e?auto=format&fit=crop&w=400&q=80"},
    {"id": "p15", "name": "Farm-Fresh Organic Fruit Box", "price": 35.00, "category": "food", "image": "https://images.unsplash.com/photo-1619566636858-adf3ef46400b?auto=format&fit=crop&w=400&q=80"},
    {"id": "p16", "name": "Napa Valley Wine Tasting Set", "price": 110.00, "category": "food", "image": "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=400&q=80"},
    
    # Online Services
    {"id": "p17", "name": "Premium Streaming Annual Plan", "price": 119.99, "category": "online_services", "image": "https://images.unsplash.com/photo-1611162616475-46b635cb6868?auto=format&fit=crop&w=400&q=80"},
    {"id": "p18", "name": "2TB Cloud Storage Renewal", "price": 99.99, "category": "online_services", "image": "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?auto=format&fit=crop&w=400&q=80"},
    {"id": "p19", "name": "Business Web Hosting Package", "price": 240.00, "category": "online_services", "image": "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=400&q=80"},
    {"id": "p20", "name": "Digital Marketing Analytics Pro", "price": 49.99, "category": "online_services", "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80"}
]

shap_explainer = None

def load_models():
    global preprocessor, supervised_model, unsupervised_model, ae_model, engineer, ensemble, db_df, shap_explainer
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
        
        print("Initializing SHAP explainer...")
        if supervised_model and supervised_model.model:
            # We use TreeExplainer for Random Forest
            shap_explainer = shap.TreeExplainer(supervised_model.model)
        
        print("System ready for real-time inference.")
        
    except Exception as e:
        print(f"Error loading models. Have you run the training pipeline yet? Error: {str(e)}")


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/shop')
@limiter.limit("5 per minute")
def shop_page():
    return render_template('shop.html')

@app.route('/api/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Legacy login endpoint — kept for backwards compat but now just verifies Firebase token."""
    decoded = verify_firebase_token(request)
    if not decoded:
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    uid = decoded['uid']
    # Read user profile from Firestore
    user_doc = firestore_db.collection('users').document(uid).get()
    if not user_doc.exists:
        return jsonify({"status": "error", "message": "User not found in Firestore"}), 404
    user_data = user_doc.to_dict()
    live_user_base[uid] = (user_data.get('home_lat', 40.7128), user_data.get('home_lon', -74.0060))
    return jsonify({"status": "success", "user": {"id": uid, "name": user_data.get('display_name', 'User')}})

import random

@app.route('/api/signup', methods=['POST'])
@limiter.limit("5 per minute")
def signup():
    """Handles new user sign up: creates Firebase Auth user & seeds Firestore profile."""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Invalid request body"}), 400
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not name or not email or not password:
            return jsonify({"status": "error", "message": "Missing fields"}), 400
        
        if len(password) < 6:
            return jsonify({"status": "error", "message": "Password must be at least 6 characters."}), 400
            
        # 1. Create in Firebase Auth
        user_record = firebase_auth.create_user(
            email=email,
            password=password,
            display_name=name
        )
        uid = user_record.uid
        
        # 2. Seed dummy profile in Firestore
        uid_label = f"U{random.randint(10000, 99999)}"
        card_prefix = random.choice(["4532","4716","5425","5168","4929","5307","5218","4024"])
        card_mid = f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
        
        profile = {
            'uid_label': uid_label,
            'email': email,
            'display_name': name,
            'balance': round(random.uniform(3000.0, 10000.0), 2),
            'card_name': name,
            'card_number': f"{card_prefix} {card_mid}",
            'card_expiry': f"{random.randint(1,12):02d}/{random.randint(26,30)}",
            'card_cvv': f"{random.randint(100,999)}",
            'billing_address': "123 New User St, Tech City",
            'address': "123 New User St, Tech City 10001",
            'home_lat': 40.7128,  # NYC default
            'home_lon': -74.0060,
            'role': 'user'
        }
        
        firestore_db.collection('users').document(uid).set(profile)
        
        return jsonify({
            "status": "success", 
            "message": "User created successfully",
            "uid": uid
        })
        
    except firebase_auth.EmailAlreadyExistsError:
        return jsonify({"status": "error", "message": "Email already exists."}), 400
    except Exception as e:
        print(f"Signup error: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify({"status": "success", "products": PRODUCTS})

@app.route('/api/user/<user_id>/payment', methods=['GET'])
def get_payment_info(user_id):
    """Return per-user payment credentials from Firestore."""
    # user_id here is the Firebase UID
    user_doc = firestore_db.collection('users').document(user_id).get()
    if not user_doc.exists:
        return jsonify({"status": "error", "message": "User not found"}), 404
    u = user_doc.to_dict()
    return jsonify({
        "status": "success",
        "payment": {
            "card_name": u.get("card_name", ""),
            "card_number": u.get("card_number", ""),
            "card_expiry": u.get("card_expiry", ""),
            "card_cvv": u.get("card_cvv", ""),
            "billing_address": u.get("billing_address", "")
        }
    })

@app.route('/api/user/<user_id>/payment', methods=['POST'])
def update_payment_info(user_id):
    """Update per-user payment credentials in Firestore."""
    decoded = verify_firebase_token(request)
    if not decoded or decoded['uid'] != user_id:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Invalid request"}), 400
        
    try:
        update_data = {
            "card_name": data.get("card_name", ""),
            "card_number": data.get("card_number", ""),
            "card_expiry": data.get("card_expiry", ""),
            "card_cvv": data.get("card_cvv", ""),
            "billing_address": data.get("billing_address", "")
        }
        firestore_db.collection('users').document(user_id).update(update_data)
        return jsonify({"status": "success", "message": "Payment method updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    global db_df
    try:
        data = request.json
        
        # Extract fields from incoming real-time request
        user_id = data.get('user_id', 'U00000')
        merchant_id = data.get('merchant_id', 'M0000')
        merchant_category = data.get('merchant_category', 'retail')
        amount = float(data.get('amount', 0.0))
        location_lat = float(data.get('location_lat', 0.0))
        location_lon = float(data.get('location_lon', 0.0))

        # ── Session behavior signals (Feature 2: Velocity Parameter Tuning) ──
        # Sent by the checkout JS — all optional, default to None = unknown
        session_fill_time    = data.get('session_fill_time')     # seconds to fill payment form
        session_keystroke_ms = data.get('session_keystroke_ms')  # avg ms between keystrokes
        session_mouse_entropy= data.get('session_mouse_entropy') # 0-1 movement randomness
        session_paste_count  = data.get('session_paste_count')   # number of paste events
        session_backspace_ratio = data.get('session_backspace_ratio')  # backspace / total keys
        # Use server-side real IP (more reliable than client-reported)
        client_ip = (
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or request.remote_addr
            or data.get('ip_address', 'UNKNOWN')
        )
        ip_address           = client_ip
        card_number_hint     = data.get('card_number_hint', '')  # last-4 only, for graph hash
        
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
        raw_tx_count = float(latest_txn['tx_count_12h'].values[0])
        
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
        adj_iso = iso_score if raw_geo_km > 500 else 0.0
        adj_ae = ae_score if raw_geo_km > 500 else 0.0
        
        base_risk = ensemble.calculate_risk_score(
            np.array([sup_prob]), 
            np.array([adj_iso]), 
            np.array([adj_ae])
        )[0]
        
        # 6. Contextual rule-based boosting
        boost = 0
        reasons = []
        
        if raw_geo_km > 500: 
            reasons.append("Unusual Location")
            boost += 20
        if raw_geo_km > 3000:
            boost += 15
            
        if raw_spend_z >= 1.4:
            reasons.append("Unusual Spend Amount")
            boost += 35
        if raw_spend_z >= 2.5:
            boost += 35
            
        # Velocity manual override for responsive demonstration
        if raw_tx_count >= 3.0:
            if "High Transaction Velocity" not in reasons:
                reasons.append("High Transaction Velocity")
            boost += 35
        if raw_tx_count >= 5.0:
            boost += 35
            
        if raw_amount > 500 and raw_geo_km > 500:
            reasons.append("High-value overseas purchase")
            boost += 15
            
        if sup_prob > 0.6:
            reasons.append("Matches known fraud patterns")

        # ── Feature 2: Session Behavior Boosts (Bot Detection) ──────────────
        bot_signals = 0

        # Signal 1: Suspiciously fast form fill (< 4s for a real human is near-impossible)
        if session_fill_time is not None and session_fill_time < 4:
            bot_signals += 1
            boost += 25

        # Signal 2: Robot-like keystroke speed (avg < 40ms between keys = automated input)
        if session_keystroke_ms is not None and session_keystroke_ms < 40:
            bot_signals += 1
            boost += 20

        # Signal 3: Linear mouse movement (entropy < 0.15 = no human variance)
        if session_mouse_entropy is not None and session_mouse_entropy < 0.15:
            bot_signals += 1
            boost += 15

        # Signal 4: All fields pasted (no typing at all = scripted bot)
        if session_paste_count is not None and session_paste_count >= 4:
            bot_signals += 1
            boost += 15

        # Signal 5: Zero backspaces across multi-field form (humans always mistype)
        if session_backspace_ratio is not None and session_backspace_ratio == 0 and session_fill_time is not None and session_fill_time < 30:
            bot_signals += 1
            boost += 10

        if bot_signals >= 2 and "Automated Bot Behavior Detected" not in reasons:
            reasons.append("Automated Bot Behavior Detected")
            print(f"DEBUG: Bot signals triggered: {bot_signals}/5")
        risk_score = min(int(base_risk) + boost, 100)
        
        decision = ensemble.get_decision(risk_score)
        
        # --- Business Logic Override: Frictionless Spend Limits ---
        # Demo adjustment: Don't block daily purchases (< $150) across the board even if transaction velocity/bot is high,
        # UNLESS the risk score is an absolute screaming red flag (>= 90).
        if amount <= 150.0 and risk_score < 90:
            decision = "ALLOW"
            reasons = []
            
        if not reasons and decision != "ALLOW":
            reasons.append("System Anomaly Detected")

        # 8. SHAP Explainability (Only for flagged transactions)
        try:
            if decision != 'ALLOW' and shap_explainer is not None:
                # Calculate SHAP values for this instance
                shap_vals = shap_explainer.shap_values(X)
                
                # For RF, shap_values might be a list where index 1 is the positive class (fraud)
                if isinstance(shap_vals, list):
                    fraud_shap = shap_vals[1][0]
                else:
                    if len(shap_vals.shape) == 3:
                        fraud_shap = shap_vals[0, :, 1]
                    else:
                        fraud_shap = shap_vals[0]
                        
                feat_importance = [(features[i], fraud_shap[i]) for i in range(len(features))]
                feat_importance.sort(key=lambda x: x[1], reverse=True)
                top_3 = [f for f in feat_importance[:3] if f[1] > 0]
                
                if top_3:
                    icon = "🚫" if decision == "BLOCK" else "⚠️"
                    action_str = "Blocked" if decision == "BLOCK" else "Flagged"
                    reasons.insert(0, f"{icon} {action_str} — Top reasons:")
                    for i, (feat, val) in enumerate(top_3):
                        reasons.insert(i + 1, f"  {i+1}. {feat}: +{val:.2f}")
        except Exception as e:
            print(f"SHAP Error: {e}")

        # --- Dispatch Notification ---
        try:
            # Look up the actual email from Firestore using the user_id (which is the document ID)
            user_doc = firestore_db.collection('users').document(user_id).get()
            if user_doc.exists:
                user_email = user_doc.to_dict().get('email')
                if user_email:
                    send_transaction_notification(user_email, amount, merchant_category, decision, reasons)
            else:
                print(f"Warning: Could not find user {user_id} for notification dispatch.")
        except Exception as notify_err:
            print(f"Warning: Failed to dispatch notification: {notify_err}")

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
        
        # ─── Persist to Firestore ───
        from firebase_admin import firestore as firestore_admin
        firestore_db.collection('transactions').document(result['transaction_id']).set({
            "transaction_id": result['transaction_id'],
            "user_id": user_id,
            "amount": amount,
            "category": merchant_category,
            "risk_score": int(risk_score),
            "decision": decision,
            "reasons": reasons,
            "supervised_confidence": round(float(sup_prob) * 100, 2),
            "anomaly_index": round(float(iso_score) * 100, 2),
            "location_lat": location_lat,
            "location_lon": location_lon,
            "timestamp": timestamp.strftime("%H:%M:%S"),
            "created_at": firestore_admin.SERVER_TIMESTAMP
        })

        # ─── Write to Neo4j graph asynchronously (non-blocking) ───
        try:
            user_doc = firestore_db.collection('users').document(user_id).get()
            display_name = user_doc.to_dict().get('display_name', user_id) if user_doc.exists else user_id
        except Exception:
            display_name = user_id

        def _neo4j_write():
            neo4j_graph.write_transaction_graph(
                user_id=user_id,
                user_name=display_name,
                tx_id=result['transaction_id'],
                amount=amount,
                category=merchant_category,
                risk_score=int(risk_score),
                decision=decision,
                ip_address=ip_address,
                card_number=card_number_hint,
            )
        threading.Thread(target=_neo4j_write, daemon=True).start()
        
        # Append to live memory so velocity/behavior ticks up for subsequent queries during the live demo
        db_df = pd.concat([db_df, txn_df], ignore_index=True)
            
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    load_models()
    # Run on port 5001 to avoid common 5000 conflicts
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
