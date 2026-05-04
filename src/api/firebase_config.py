"""
Firebase Admin SDK initialization — shared by app.py and bank_app.py.
"""
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth

# Path to the service account key JSON
_KEY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../ai-financial-fraud-detection-firebase-adminsdk-fbsvc-c690380498.json')
)

# Initialize only once (both app.py and bank_app.py import this)
if not firebase_admin._apps:
    cred = credentials.Certificate(_KEY_PATH)
    firebase_admin.initialize_app(cred)

# Expose a Firestore client
db = firestore.client()

def verify_firebase_token(request):
    """
    Extract and verify the Firebase ID token from the Authorization header.
    Returns the decoded token dict on success, or None on failure.
    """
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    token = auth_header.split('Bearer ')[1]
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        return None
