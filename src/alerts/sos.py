import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load the hidden credentials from your .env file
load_dotenv()

def send_sos_alert(driver_name="Driver"):
    """Sends an emergency SMS via Twilio when critical drowsiness is detected."""
    
    # Securely fetch the API keys
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_number = os.getenv('TWILIO_PHONE_NUMBER')
    destination_number = os.getenv('DESTINATION_PHONE_NUMBER')

    # Safety check: Prevent crashes if the .env file is empty
    if not all([account_sid, auth_token, twilio_number, destination_number]):
        print("⚠️ Twilio credentials missing. Check your .env file!")
        return False

    try:
        # Connect to Twilio
        client = Client(account_sid, auth_token)
        
        # Draft and send the message
        message = client.messages.create(
            body=f"🚨 URGENT: Critical Drowsiness/Distraction detected for {driver_name}. Please check in immediately.",
            from_=twilio_number,
            to=destination_number
        )
        
        print(f"✅ SOS SMS Sent Successfully! Message SID: {message.sid}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send SMS. Error: {e}")
        return False

# Quick tester block! This only runs if you execute this specific file.
if __name__ == "__main__":
    print("Testing Twilio SOS Module...")
    send_sos_alert("Test Driver")