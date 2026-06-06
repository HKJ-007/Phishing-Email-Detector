import pandas as pd  
import numpy as np  
import re  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,  
                             precision_score, recall_score, f1_score, roc_auc_score)  
import matplotlib.pyplot as plt  
import joblib  
  
df = pd.read_csv('phishing_emails.csv')  
print(f"Loaded {len(df)} email samples (100 phishing + 100 legitimate)\n")  
  
def extract_features(text):  
    original_text = str(text)  
    text_lower = original_text.lower()  
    features = {}  
       
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_lower)  
    features['num_urls'] = len(urls)  
    features['has_url'] = 1 if features['num_urls'] > 0 else 0  
    features['contains_ip'] = 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text_lower) else 0  
      
    suspicious = ['click', 'verify', 'suspended', 'urgent', 'immediately', 'prize', 'won', 'claim',   
                  'login', 'password', 'alert', 'security', 'update', 'account', 'bank', 'fraud']  
    features['suspicious_count'] = sum(1 for word in suspicious if word in text_lower)  
        
    urgency = ['urgent', 'immediately', 'now', 'expired', 'suspended', 'locked', 'limited time',   
               'act now', 'warning', 'alert']  
    features['urgency_count'] = sum(1 for word in urgency if word in text_lower)  
       
    features['num_exclamation'] = original_text.count('!')  
    features['contains_currency'] = 1 if re.search(r'[\$€£¥]', original_text) else 0  
    features['capital_ratio'] = sum(1 for c in original_text if c.isupper()) / (len(original_text) + 1)  
    features['num_questions'] = original_text.count('?')  
      
    return pd.Series(features)  
    
feature_df = df['email_text'].apply(extract_features)  
  
vectorizer = TfidfVectorizer(max_features=400, stop_words='english', ngram_range=(1, 2))  
X_tfidf = vectorizer.fit_transform(df['email_text'])  
  
X = pd.concat([pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out()),   
               feature_df.reset_index(drop=True)], axis=1)  
y = df['label']  
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)  
  
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')  
model.fit(X_train, y_train)  
  
y_pred = model.predict(X_test)  
y_pred_proba = model.predict_proba(X_test)[:, 1]  
  
print("=== Model Performance ===")  
print(f"Accuracy      : {accuracy_score(y_test, y_pred):.2%}")  
print(f"Precision     : {precision_score(y_test, y_pred):.2%}")  
print(f"Recall        : {recall_score(y_test, y_pred):.2%}")  
print(f"F1 Score      : {f1_score(y_test, y_pred):.2%}")  
print(f"ROC AUC       : {roc_auc_score(y_test, y_pred_proba):.3f}\n")  
  
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))  
    
cm = confusion_matrix(y_test, y_pred)  
plt.figure(figsize=(6, 5))  
plt.imshow(cm, interpolation='nearest', cmap='Blues')  
plt.title('Phishing Detection Confusion Matrix')  
plt.colorbar()  
tick_marks = np.arange(2)  
plt.xticks(tick_marks, ['Safe', 'Phishing'])  
plt.yticks(tick_marks, ['Safe', 'Phishing'])  
  
thresh = cm.max() / 2.  
for i in range(cm.shape[0]):  
    for j in range(cm.shape[1]):  
        plt.text(j, i, format(cm[i, j], 'd'),  
                 ha="center", va="center",  
                 color="white" if cm[i, j] > thresh else "black")  
  
plt.ylabel('True label')  
plt.xlabel('Predicted label')  
plt.tight_layout()  
plt.savefig('confusion_matrix.png')  
print("Confusion matrix saved as 'confusion_matrix.png'")  
    
joblib.dump(model, 'phishing_model.pkl')  
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  
print("Model and vectorizer saved.")  
  
def predict_email(email_text):  
    features = extract_features(email_text)  
    text_vec = vectorizer.transform([email_text])  
    combined = pd.concat([pd.DataFrame(text_vec.toarray(), columns=vectorizer.get_feature_names_out()),   
                          pd.DataFrame([features])], axis=1)  
    pred = model.predict(combined)[0]  
    prob = model.predict_proba(combined)[0][pred]  
    label = "Phishing" if pred == 1 else "Safe"  
    return label, prob * 100  
  
print("\n=== Sample Predictions ===")  
test_emails = [  
    "Your account is suspended. Verify now: http://fakebank.com/login",  
    "Team meeting notes from yesterday are attached. Please review by Friday.",  
    "URGENT: You've won $5000! Claim your prize immediately at http://winner-alert.com",  
    "Monthly project update: All deadlines met. Great work team!"  
]  
  
for email in test_emails:  
    result, confidence = predict_email(email)  
    print(f"→ {result} ({confidence:.1f}%) | {email[:65]}...")