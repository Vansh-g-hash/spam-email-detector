# 📩 Spam Email Detector (ML + NLP + Gradio)

A Machine Learning project to classify emails/messages as **Spam** or **Ham (Not Spam)** using NLP techniques and TF-IDF vectorization.

## 🚀 Features
- Preprocessing: lowercasing, cleaning, TF-IDF
- Models: Naive Bayes 🤖, Logistic Regression 📈, SVM ⚡
- Visualizations:
  - Spam vs Ham distribution
  - WordCloud of spam messages
  - Confusion Matrices
- Interactive Gradio Web App
- Leaderboard with model accuracy 🏆

## 📊 Results
| Model                 | Accuracy |
|------------------------|----------|
| Naive Bayes 🤖         | ~97%     |
| Logistic Regression 📈 | ~98%     |
| SVM ⚡                 | ~98%     |

## 🖥️ Installation
```bash
git clone https://github.com/your-username/spam-email-detector.git
cd spam-email-detector
pip install -r requirements.txt
python app.py
