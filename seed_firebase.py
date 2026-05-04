"""
Seed Firebase Auth + Firestore with 20 shop users, 1 bank admin,
and ~200 historical transactions.

Run once:  python seed_firebase.py
Re-run to reset all data.
"""
import os
import sys
import random
import datetime

import firebase_admin
from firebase_admin import credentials, auth, firestore

# ── Init Firebase Admin ──
KEY_PATH = os.path.join(os.path.dirname(__file__),
    'ai-financial-fraud-detection-firebase-adminsdk-fbsvc-c690380498.json')
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── User data (same 20 users from the original app.py) ──
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
CARD_PREFIXES = ["4532","4716","5425","5168","4929","4539","5307","4485","5218","4024",
                 "4556","5294","4916","5102","4738","5387","4652","5043","4831","5276"]
STREETS = [
    "381 Park Ave","742 Elm St","55 Broadway","128 Oak Blvd","900 Pine Rd",
    "217 Maple Dr","463 Cedar Ln","88 Birch Way","1550 Walnut St","320 Spruce Ave",
    "611 Main St","77 River Rd","203 Lake Dr","445 Hill Ct","160 Valley Rd",
    "892 Ocean Blvd","34 Summit Ave","510 Forest Ln","275 Bridge St","1020 Harbor Dr"
]
CITIES = [
    "New York, NY","San Francisco, CA","Chicago, IL","Houston, TX","Phoenix, AZ",
    "Philadelphia, PA","San Antonio, TX","San Diego, CA","Dallas, TX","Austin, TX",
    "Jacksonville, FL","Columbus, OH","Charlotte, NC","Indianapolis, IN","Seattle, WA",
    "Denver, CO","Boston, MA","Nashville, TN","Portland, OR","Las Vegas, NV"
]
ZIPCODES = [
    "10001","94102","60601","77001","85001",
    "19101","78201","92101","75201","73301",
    "32099","43004","28201","46201","98101",
    "80201","02101","37201","97201","89101"
]
CATEGORIES = ["high_risk_electronics","retail","food","travel","online_services"]
DECISIONS  = ["ALLOW","ALLOW","ALLOW","ALLOW","ALLOW","ALLOW","ALLOW","ALLOW","FLAG_FOR_REVIEW","BLOCK"]

# ── 0. Delete old users from Firebase Auth ──
print("=== Cleaning up old Firebase Auth users ===")
try:
    page = auth.list_users()
    uids_to_delete = [u.uid for u in page.users]
    while page.next_page_token:
        page = auth.list_users(page_token=page.next_page_token)
        uids_to_delete.extend([u.uid for u in page.users])
    if uids_to_delete:
        result = auth.delete_users(uids_to_delete)
        print(f"  🗑️  Deleted {result.success_count} users, {result.failure_count} failures")
    else:
        print("  No existing users to delete")
except Exception as e:
    print(f"  Cleanup error: {e}")

# ── 1. Create Firebase Auth users + Firestore user docs ──
print("\n=== Creating Firebase Auth users (firstname.lastname@shopvault.com) ===")

for i in range(1, 21):
    idx = i - 1
    uid_label = f"U{i:05d}"
    first = FIRST_NAMES[idx].lower()
    last = LAST_NAMES[idx].lower()
    email = f"{first}.{last}@shopvault.com"
    display_name = f"{FIRST_NAMES[idx]} {LAST_NAMES[idx]}"
    card_mid = f"{1000+i*111:04d} {2000+i*73:04d} {3000+i*37:04d}"
    address = f"{STREETS[idx]}, {CITIES[idx]} {ZIPCODES[idx]}"

    try:
        user_record = auth.create_user(
            email=email,
            password='password123',
            display_name=display_name
        )
        uid = user_record.uid
        print(f"  ✅ {email:40s} ({display_name})")
    except auth.EmailAlreadyExistsError:
        user_record = auth.get_user_by_email(email)
        uid = user_record.uid
        print(f"  ⏭️  {email:40s} already exists")

    # Write Firestore user doc keyed by UID
    db.collection('users').document(uid).set({
        'uid_label': uid_label,
        'email': email,
        'display_name': display_name,
        'balance': round(5000.0 + i * 500, 2),
        'card_name': display_name,
        'card_number': f"{CARD_PREFIXES[idx]} {card_mid}",
        'card_expiry': f"{((i*3)%12)+1:02d}/{26+(i%4)}",
        'card_cvv': f"{100+i*37:03d}",
        'billing_address': f"{STREETS[idx]}, {CITIES[idx]}",
        'address': address,
        'home_lat': 40.7128,
        'home_lon': -74.0060,
        'role': 'user'
    })

# Create admin user
print("\n=== Creating bank admin ===")
try:
    admin_record = auth.create_user(
        email='admin@shopvault.com',
        password='admin123',
        display_name='Bank Admin'
    )
    admin_uid = admin_record.uid
    print(f"  ✅ admin@shopvault.com")
except auth.EmailAlreadyExistsError:
    admin_record = auth.get_user_by_email('admin@shopvault.com')
    admin_uid = admin_record.uid
    print(f"  ⏭️  admin@shopvault.com already exists")

db.collection('users').document(admin_uid).set({
    'email': 'admin@shopvault.com',
    'display_name': 'Bank Admin',
    'address': 'ShopVault HQ, New York, NY 10001',
    'role': 'admin'
})

# ── 2. Seed historical transactions ──
print("\n=== Seeding 200 transactions ===")

all_users = []
for i in range(1, 21):
    first = FIRST_NAMES[i-1].lower()
    last = LAST_NAMES[i-1].lower()
    email = f"{first}.{last}@shopvault.com"
    u = auth.get_user_by_email(email)
    all_users.append({'uid': u.uid, 'label': f"U{i:05d}"})

batch = db.batch()
now = datetime.datetime.now(datetime.timezone.utc)

for t in range(200):
    user = random.choice(all_users)
    cat = random.choice(CATEGORIES)
    amount = round(random.uniform(10, 2000), 2)
    risk = random.randint(5, 55)
    decision = random.choice(DECISIONS)
    ts = now - datetime.timedelta(hours=random.randint(1, 720))

    doc_ref = db.collection('transactions').document(f"SEED_{t:04d}")
    batch.set(doc_ref, {
        'transaction_id': f"SEED_{t:04d}",
        'user_id': user['label'],
        'user_uid': user['uid'],
        'amount': amount,
        'category': cat,
        'risk_score': risk,
        'decision': decision,
        'reasons': [],
        'supervised_confidence': round(random.uniform(0.1, 0.9), 2),
        'anomaly_index': round(random.uniform(0.05, 0.6), 2),
        'location_lat': 40.7128 + random.uniform(-0.05, 0.05),
        'location_lon': -74.0060 + random.uniform(-0.05, 0.05),
        'timestamp': ts.isoformat(),
        'created_at': ts
    })

batch.commit()
print(f"  ✅ Seeded 200 transactions")

print("\n🎉 Done! Firebase is ready.")
print(f"   Shop login:  alex.johnson@shopvault.com / password123")
print(f"   Bank login:  admin@shopvault.com / admin123")
print(f"\n   All 20 users: firstname.lastname@shopvault.com / password123")
