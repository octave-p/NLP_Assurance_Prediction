# 🛡️ NLP Insurance Reviews Analysis & RAG Assistant

An end-to-end Natural Language Processing project designed to analyze 34,000+ customer reviews from the French insurance sector. The project includes data engineering, target modeling (Sentiment Analysis), comparison of classical and deep learning models, and the deployment of a fully interactive Streamlit web application featuring an extractive RAG (Retrieval-Augmented Generation) assistant.

## 🎯 Project Objectives
- **Data Pipeline:** Clean, normalize, and translate raw customer reviews to build a robust NLP dataset.
- **Unsupervised Learning:** Discover hidden semantic structures using Topic Modeling (NMF) and Word Embeddings (Word2Vec).
- **Supervised Learning:** Compare the performance of three distinct NLP architectures for Sentiment Analysis.
- **Web Application:** Deploy the best model in a Streamlit app to provide real-time predictions, explainability, and a QA assistant.

## 🧠 Models Comparison (Sentiment Analysis)
We framed the problem as a 3-class Sentiment Analysis (Positive, Neutral, Negative) and benchmarked the following models:
1. **TF-IDF + Logistic Regression (Baseline):** Accuracy: **0.75** / Macro F1: **0.67** 🏆
2. **CamemBERT (Feature Extraction):** Accuracy: **0.72**
3. **Keras Deep Learning (Custom Embedding Layer):** Accuracy: **0.66**

*Note: The TF-IDF baseline outperformed Deep Learning models due to the highly polarized nature of the vocabulary in customer reviews (e.g., "scam", "perfect"), where statistical approaches excel. Moreover, custom embeddings (Keras) lacked the massive data volume required to rival pre-trained methods.*

## 🚀 Streamlit Application Features
The application (`app.py`) is built to be fast, local, and fully interactive:
- **📊 Insurer Analysis (Summary):** Dynamic generation of average ratings and distribution charts per insurance company.
- **🔍 Information Retrieval:** Advanced keyword and entity filtering to explore specific customer verbatims.
- **🔮 Prediction & Explainability:** Real-time sentiment prediction of user inputs, highlighting the specific keywords that drove the model's decision.
- **🤖 Virtual Assistant (Extractive RAG & QA):** A hybrid Q&A system that detects entities and intents (e.g., Pricing, Claims) in user queries, retrieves relevant context from the database, calculates specific metrics, and extracts raw source verbatims to prevent LLM hallucinations.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **NLP & Embeddings:** NLTK, Scikit-learn (TF-IDF, NMF), Gensim (Word2Vec)
- **Deep Learning:** TensorFlow / Keras, HuggingFace `sentence-transformers` (CamemBERT)
- **Web App:** Streamlit

## 📊 Project Presentation

The project slideshow is available in [`presentation.html`](presentation.html) — download the file and open it in a browser to view it.

## 🎬 Project Video

> **[▶ Watch the project presentation video](#)**
> *(Link to be updated)*

---

## ⚙️ How to run the app locally
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```
