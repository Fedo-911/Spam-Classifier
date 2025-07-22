# 📧 Email Spam Classifier

A machine learning project that classifies email/SMS messages as **spam** or **not spam** using natural language processing (NLP) and the Multinomial Naive Bayes algorithm. This project is built in **Google Colab** and deployed with **Streamlit**.



## 🚀 Demo

🌐 Live App: https://spam-classifier-fardeen.streamlit.app/



## 📁 Dataset

The dataset used is the **[SMSSpamCollection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)** from the UCI Machine Learning Repository.

- 5,574 SMS messages
- Two classes: `ham` (not spam) and `spam`



## 🛠️ Features

- Text preprocessing with TF-IDF Vectorization
- Spam detection using Multinomial Naive Bayes
- Clean, minimal Streamlit web app
- Model and vectorizer serialized with `joblib`



## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
   
 2. Install dependencies:
    
    pip install -r requirements.txt

3. Run locally:

   streamlit run app.py


   
## 🔍 File Structure
spam-classifier/

├── app.py                   # Streamlit web app

├── spam_model.pkl           # Trained ML model

├── tfidf_vectorizer.pkl     # TF-IDF vectorizer

├── requirements.txt         # Python dependencies

├── runtime.txt              # Python version (for Streamlit Cloud)

└── README.md                # Project overview

## 📊 Model Performance
| Metric    | Score |
| --------- | ----- |
| Accuracy  |~98% |
| Precision | High  |
| Recall    | High  |

## ☁️ Deploy on Streamlit Cloud
1. Push this project to a GitHub repo.

2. Go to streamlit.io/cloud → "New App".

3. Connect your GitHub and select the repo.

4. Use app.py as the entry point.

5. Done! 🎉

## 📚 Requirements
• Python 3.10+

• Streamlit

• Scikit-learn

• Pandas

• NumPy

• Joblib

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

## 📜 License
This project is open source and available under the MIT License.

## 💡 Inspiration
Built as part of a machine learning learning project to explore NLP, spam detection, and deploying ML models with minimal infrastructure.
