# app.py

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ===============================
# 1. Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# ===============================
# 2. Text Cleaning
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['message'] = df['message'].apply(clean_text)

# ===============================
# 3. Train Models
# ===============================
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB().fit(X_train_tfidf, y_train)
log_model = LogisticRegression(max_iter=2000).fit(X_train_tfidf, y_train)
svm_model = LinearSVC().fit(X_train_tfidf, y_train)

# Accuracy scores
acc_scores = {
    "Naive Bayes 🤖": accuracy_score(y_test, nb_model.predict(X_test_tfidf)),
    "Logistic Regression 📈": accuracy_score(y_test, log_model.predict(X_test_tfidf)),
    "SVM ⚡": accuracy_score(y_test, svm_model.predict(X_test_tfidf))
}

# ===============================
# 4. Streamlit UI
# ===============================
st.title("📩 Spam Email Detector")
st.write("Built with **NLP + ML (Naive Bayes, Logistic Regression, SVM)**")

# Sidebar
st.sidebar.title("📊 Model Performance")
st.sidebar.table(pd.DataFrame(acc_scores.items(), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False))

# WordCloud
st.subheader("☁️ WordCloud of Spam Messages")
spam_words = " ".join(df[df['label']==1]['message'])
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="spring").generate(spam_words)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# Confusion Matrix (Logistic Regression)
st.subheader("🔎 Confusion Matrix (Logistic Regression)")
y_pred = log_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"], ax=ax)
st.pyplot(fig)

# Prediction
st.subheader("✍️ Test Your Message")
user_input = st.text_area("Enter a message:")
if st.button("Predict"):
    user_clean = clean_text(user_input)
    user_tfidf = vectorizer.transform([user_clean])

    preds = {
        "Naive Bayes 🤖": "🚨 Spam" if nb_model.predict(user_tfidf)[0] else "✅ Ham",
        "Logistic Regression 📈": "🚨 Spam" if log_model.predict(user_tfidf)[0] else "✅ Ham",
        "SVM ⚡": "🚨 Spam" if svm_model.predict(user_tfidf)[0] else "✅ Ham"
    }

    st.write("### Predictions:")
    st.json(preds)
