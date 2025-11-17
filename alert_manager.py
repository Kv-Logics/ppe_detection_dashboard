import smtplib                   
from email.mime.text import MIMEText 

def send_email_alert(recipient_email, sender_email, app_password, violation_details):
    """
    Sends a high-severity violation alert via Gmail SMTP using a Google App Password.
    """
    try:
        if not app_password or not sender_email or not recipient_email:
            print("Email alert skipped: Credentials or recipient missing.")
            return False

        # Construct the email body
        subject = f"HIGH PRIORITY VIOLATION: {violation_details['violation_type']}"
        body = f"""
        --- PPE VIOLATION ALERT ---

        Timestamp: {violation_details['timestamp']}
        Violation Type: {violation_details['violation_type']}
        Severity: {violation_details['severity']}
        Confidence: {round(violation_details['confidence'] * 100, 2)}%
        
        Action Required: Review the dashboard logs immediately.
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email

        # Gmail SMTP Server details
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        
        # Connect, secure, login, and send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        # Login using the Google App Password
        server.login(sender_email, app_password)
        
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        
        print(f"Email alert sent to {recipient_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("SMTP Authentication Failed: Check Sender Email and Google App Password.")
        return False
    except Exception as e:
        print(f"Failed to send alert email: {e}")
        return False