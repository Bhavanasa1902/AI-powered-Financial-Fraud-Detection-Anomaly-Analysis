import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Simple .env parser to avoid needing python-dotenv
env_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

def send_transaction_notification(email, amount, merchant, decision, reasons):
    """
    Sends an email notification for a transaction evaluation.
    Requires SMTP_EMAIL and SMTP_PASSWORD environment variables for real dispatch.
    Falls back to console printing for demo/debugging if no credentials are set.
    """
    if not email:
        return
        
    subject = f"ShopVault Alert: Transaction {decision}"
    status_text = "APPROVED" if decision == "ALLOW" else decision.replace("_", " ")
    
    body = f"""Hello,

A new transaction has been processed on your ShopVault account:

Amount: ${amount:.2f}
Merchant: {merchant}
Decision: {status_text}
"""
    if reasons and decision != "ALLOW":
        body += f"\nReasons flagged: {', '.join(reasons)}"
        
    body += "\n\nIf you did not make this transaction, please contact support immediately.\n\nShopVault Security Team"
    
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    
    has_credentials = smtp_email and smtp_password
    
    # Always print clearly for the demo
    print("\n" + "="*50)
    print(f"📧 [NOTIFICATION MANAGER]")
    if has_credentials:
        print(f"Attempting to send real email to: {email}")
    else:
        print(f"MOCK DISPATCH (No SMTP credentials found in .env)")
        print(f"To: {email}")
    print(f"Subject: {subject}")
    print(f"Body: {body.strip()}")
    print("="*50 + "\n")
    
    if not has_credentials:
        return
        
    # Send actual email
    try:
        msg = MIMEMultipart()
        msg['From'] = str(smtp_email)
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(str(smtp_email), str(smtp_password))
        server.send_message(msg)
        server.quit()
        print(f"✅ Successfully sent email to {email}")
    except Exception as e:
        print(f"❌ Failed to send email to {email}: {str(e)}")
