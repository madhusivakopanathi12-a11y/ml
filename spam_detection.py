import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def clean_email_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_dataset(csv_path: str = "emails.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {"text", "spam"}.issubset(df.columns):
            return df[["text", "spam"]].dropna()
        if {"v1", "v2"}.issubset(df.columns):  # Kaggle / UCI SMS & Spam format
            mapped = df[["v2", "v1"]].rename(columns={"v2": "text", "v1": "spam"})
            mapped["spam"] = mapped["spam"].map({"spam": 1, "ham": 0})
            return mapped.dropna()

    # Built-in sample dataset (replace or expand with a real CSV dataset for higher accuracy)
    rows = [
        ("URGENT: Your Gmail account will be suspended! Verify now at http://fake-login.xyz", 1),
        ("Congratulations! You won a $1,000 Amazon gift card. Claim prize today!", 1),
        ("Earn $5000 weekly working from home with zero experience. Click here now!", 1),
        ("Limited time crypto investment offer: double your bitcoin in 24 hours", 1),
        ("You have been selected for a free iPhone 16 Pro. Reply with your details", 1),
        ("Act fast! Cheap loans approved in minutes regardless of credit score", 1),
        ("Winner alert! Claim your lottery cash reward before it expires tonight", 1),
        ("Exclusive weight loss pill deal 80% off. Order today for free shipping", 1),
        ("Security alert: Unusual sign-in attempt blocked. Reset password immediately", 1),
        ("Get rich quick trading forex signals from our VIP Telegram group", 1),
        ("Hi team, attaching the project slides for tomorrow's 10 AM standup meeting.", 0),
        ("Hey Alex, are we still grabbing lunch near the office around 1 PM?", 0),
        ("Your electricity bill receipt for November is available in your account.", 0),
        ("Can you review the pull request on GitHub when you get a chance today?", 0),
        ("Reminder: Dentist appointment scheduled for Friday at 4:30 PM.", 0),
        ("Thanks for sharing the notes from the lecture, super helpful!", 0),
        ("Please find the updated invoice for last month's cloud hosting attached.", 0),
        ("Let's catch up over a call this weekend to plan the road trip.", 0),
        ("Your flight itinerary and boarding pass details are confirmed below.", 0),
        ("Meeting rescheduled to Thursday afternoon; calendar invite updated.", 0),
    ]
    return pd.DataFrame(rows, columns=["text", "spam"])


def train_spam_classifier(df: pd.DataFrame) -> Pipeline:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_email_text)
    X = df["clean_text"]
    y = df["spam"].astype(int)

    stratify_arg = y if y.value_counts().min() >= 2 and len(df) >= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    return model


def predict_email(model: Pipeline, email_text: str) -> None:
    cleaned = clean_email_text(email_text)
    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    label = "SPAM" if pred == 1 else "NOT SPAM (HAM)"
    confidence = proba[pred] * 100
    print(f"[{label}] ({confidence:.1f}% confidence) -> {email_text}")


def main() -> None:
    df = get_dataset("emails.csv")
    print(f"Loaded {len(df)} emails ({df['spam'].sum()} spam, {(df['spam'] == 0).sum()} ham)")
    model = train_spam_classifier(df)

    sample_emails = [
        "Congrats! You won a free cash voucher. Click http://claim-prize.example now",
        "Hi Priya, can you send over the final dataset before 5 PM?",
        "Your account password expires in 2 hours, verify immediately",
    ]
    print("\n--- Sample Predictions ---")
    for msg in sample_emails:
        predict_email(model, msg)


if __name__ == "__main__":
    main()
