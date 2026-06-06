import pandas as pd
import random

phishing_templates = [
    "Your account has been suspended. Click here to verify immediately.",
    "URGENT: Login now to avoid account closure.",
    "You have won $5000. Claim your prize now!",
    "Security alert. Verify your bank account immediately.",
    "Update your password using this link.",
    "Your mailbox is full. Login to restore access.",
    "Suspicious activity detected. Confirm your identity.",
    "Act now! Your account will expire today.",
    "Bank security warning. Verify your credentials.",
    "Limited time offer! Claim your reward now."
]

legitimate_templates = [
    "Meeting scheduled for tomorrow at 10 AM.",
    "Please find the attached project report.",
    "Lunch with the team has been moved to Friday.",
    "Your leave request has been approved.",
    "Monthly performance review attached.",
    "Reminder about the department meeting.",
    "Project submission deadline is next week.",
    "Thank you for attending today's workshop.",
    "Please review the updated documentation.",
    "The invoice has been processed successfully."
]

data = []

for _ in range(100):
    data.append([random.choice(phishing_templates), 1])

for _ in range(100):
    data.append([random.choice(legitimate_templates), 0])

random.shuffle(data)

df = pd.DataFrame(data, columns=["email_text", "label"])

df.to_csv("phishing_emails.csv", index=False)

print("Dataset generated successfully!")
print(f"Total samples: {len(df)}")