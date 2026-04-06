# FakeShield — AI-Powered Fake News Detection System

## 📌 Project Description
FakeShield is a real-time fake news detection system built using Transformer-based deep learning (DistilBERT) to classify news articles into REAL, FAKE, or UNCERTAIN.

The system integrates NLP, automated RSS ingestion, backend APIs, and a live dashboard to combat misinformation effectively.

-----------------------------------------------------------------------------------------------------------------------

## ⚡ Key Features
- Real-time news classification  
- Monitors 8 Indian RSS feeds every 5 minutes  
- Live dashboard with WebSocket updates  
- REST API for batch predictions (up to 50 articles)  
- Transformer-based contextual understanding  

-----------------------------------------------------------------------------------------------------------------------

## 🧠 Model Details
- Model: DistilBERT (66M parameters)  
- Dataset: WELFake Dataset  
- Accuracy: 93.2%  
- F1 Score: 0.9319  
- ROC-AUC: 0.98  
- Fake Recall: 0.98  

-----------------------------------------------------------------------------------------------------------------------

## 🏗️ Architecture

```mermaid
flowchart TD
A[RSS Feeds] --> B[Scheduler]
B --> C[Preprocessing]
C --> D[DistilBERT]
D --> E[Predictions]
E --> F[Database]
F --> G[Backend]
G --> H[API & Dashboard]
```

-----------------------------------------------------------------------------------------------------------------------

## 🛠️ Tech Stack
Python, PyTorch, HuggingFace Transformers, Flask, Flask-SocketIO, SQLite, APScheduler

-----------------------------------------------------------------------------------------------------------------------

## 📂 Project Structure

fakeshield/
│── models/
│── data/
│── src/
│── app.py
│── requirements.txt

-----------------------------------------------------------------------------------------------------------------------

## ▶️ Getting Started

git clone https://github.com/adityavalaki/fakeshield.git  
cd fakeshield  
pip install -r requirements.txt  
python app.py  

-----------------------------------------------------------------------------------------------------------------------
## 🔌 API Usage

POST /predict
Request:
{
  "articles": ["news text"]
}
Response:
{
  "predictions": ["REAL"]
}

-----------------------------------------------------------------------------------------------------------------------

## 🚀 Future Improvements
- LLM upgrades  
- Cloud deployment  
- Explainable AI  

-----------------------------------------------------------------------------------------------------------------------

## ⭐ Support
Give it a star on GitHub if useful.
