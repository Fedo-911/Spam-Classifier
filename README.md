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

## 🧠 Model Training

The spam classification model was trained in **Google Colab** using the [SMSSpamCollection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from the UCI Machine Learning Repository.

All training steps are included in the notebook:  
📓 [`spam_classifier.ipynb`](notebooks/spam_classifier.ipynb)

### What’s Inside the Notebook:
1. **Dataset Download**: Downloads and loads the raw dataset from UCI.
2. **Preprocessing**: Cleans and converts the labels (`ham` → 0, `spam` → 1).
3. **Feature Extraction**: Uses `TfidfVectorizer` for numerical text representation.
4. **Model Training**: Trains a `Multinomial Naive Bayes` model.
5. **Evaluation**: Measures accuracy, precision, recall, and F1-score.
6. **Model Export**: Saves both model and vectorizer as `.pkl` files using `joblib`.

You can open and run this notebook in Google Colab to:
- Retrain the model on the same dataset
- Modify preprocessing or model type
- Tune parameters
- Save new versions for deployment


## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Fedo-911/Spam-Classifier.git
   cd Spam-Classifier
   
 2. **Install dependencies:**
    
    pip install -r requirements.txt

3. **Run locally:**

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
