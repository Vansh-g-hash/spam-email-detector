# spam_detection.py

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gradio as gr

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ===============================
# 2. Text Cleaning
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['message'] = df['message'].apply(clean_text)

# ===============================
# 3. Visualizations (before training)
# ===============================
# Class distribution
plt.figure(figsize=(5,4))
sns.countplot(x=df['label'], palette="Set2")
plt.title("📊 Spam vs Ham Distribution (0=Ham, 1=Spam)")
plt.show()

# WordCloud for Spam messages
spam_words = " ".join(df[df['label']==1]['message'])
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="spring").generate(spam_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("☁️ Most Common Words in Spam Messages")
plt.show()

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ===============================
# 5. Train Models
# ===============================
nb_model = MultinomialNB().fit(X_train_tfidf, y_train)
log_model = LogisticRegression(max_iter=2000).fit(X_train_tfidf, y_train)
svm_model = LinearSVC().fit(X_train_tfidf, y_train)

# ===============================
# 6. Evaluation + Confusion Matrix
# ===============================
model_scores = {}

def evaluate_model(model, name):
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc

    print(f"\n===== {name} Results =====")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

evaluate_model(nb_model, "Naive Bayes 🤖")
evaluate_model(log_model, "Logistic Regression 📈")
evaluate_model(svm_model, "SVM ⚡")

# Convert leaderboard dict to dataframe for display
leaderboard_df = pd.DataFrame(list(model_scores.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
print("\n🏆 Model Leaderboard:")
print(leaderboard_df)

# ===============================
# 7. Gradio App (Cool UI + Leaderboard)
# ===============================
def predict_spam_all(message):
    message_cleaned = clean_text(message)
    message_tfidf = vectorizer.transform([message_cleaned])

    preds = {
        "Naive Bayes 🤖": "🚨 Spam" if nb_model.predict(message_tfidf)[0] else "✅ Ham",
        "Logistic Regression 📈": "🚨 Spam" if log_model.predict(message_tfidf)[0] else "✅ Ham",
        "SVM ⚡": "🚨 Spam" if svm_model.predict(message_tfidf)[0] else "✅ Ham"
    }
    return preds, leaderboard_df

iface = gr.Interface(
    fn=predict_spam_all,
    inputs=gr.Textbox(lines=2, placeholder="✍️ Type a message like 'Win a free iPhone!'"),
    outputs=[gr.JSON(label="Predictions"), gr.Dataframe(label="🏆 Model Leaderboard")],
    title="📩 Spam Email Detector (ML Showcase)",
    description="⚡ Compare predictions from Naive Bayes, Logistic Regression, and SVM.\n\nMade with ❤️ by Vansh Gupta",
    theme="default"
)

iface.launch()
