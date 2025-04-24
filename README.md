# email_classifier_masker
# 📧 Email Support Classifier & PII Masking System

A production-ready system that detects and masks PII (Personally Identifiable Information) in support emails and classifies them into relevant categories using a fine-tuned DistilBERT model.

## 🚀 Features

- Detects and masks PII (names, organizations, emails, locations, phone numbers)
- Classifies emails into categories like "Billing", "Account", "Urgent", or "General"
- Provides a REST API built with FastAPI
- Interactive Streamlit frontend for real-time usage
- Robust logging and error handling

---

## 📁 Project Structure

```
.
├── api.py                  # Lightweight FastAPI endpoint
├── app.py                  # Full-featured API + Streamlit frontend
├── main.py                 # Minimal API version
├── model.py                # Model training and inference
├── pipeline.py             # PII masking + classification pipeline
├── utils.py                # SpaCy PII masking + data preprocessing
├── requirements.txt        # Python dependencies
```

---

## 🧠 Model

- **Base Model**: `distilbert-base-uncased`
- Fine-tuned using Hugging Face Transformers on masked support emails
- Classes: customizable based on labeled dataset (default: Billing, Account, Urgent, General)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/email-support-classifier.git
cd email-support-classifier
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🧪 Training the Model

Update your dataset path and run:

```python
from src.model import train_model

train_model(data_path="data/your_dataset.csv", model_path="models/bert_classifier")
```

Ensure the CSV contains:
- `email`: raw email text
- `type`: category labels

---

## 🌐 Running the API

### Option 1: Run FastAPI only

```bash
uvicorn main:app --reload --port 8000
```

### Option 2: Run FastAPI + Streamlit together

```bash
python app.py
```

---

## 🧪 API Endpoints

### `GET /`
Returns a welcome message.

### `GET /health`
Checks if SpaCy and model pipeline are loaded correctly.

### `POST /classify`
Classifies email and masks PII entities.

#### Request Body:
```json
{
  "email_text": "Hi, I'm John from Amazon. Please reset my password."
}
```

#### Response:
```json
{
  "success": true,
  "input_email_body": "Hi, I'm John from Amazon...",
  "list_of_masked_entities": [
    {"position": [9, 13], "classification": "PERSON", "entity": "John"},
    {"position": [19, 25], "classification": "ORG", "entity": "Amazon"}
  ],
  "masked_email": "Hi, I'm [PERSON] from [ORG]...",
  "category_of_the_email": "Account"
}
```

---

## 🖥️ Streamlit UI

After starting the app via `python app.py`, open:

```
http://localhost:8501
```

Paste your email and get:
- Masked version of the text
- Detected PII entities
- Classified category

---

## 🛠️ Environment Variables

You can optionally set:

```bash
export MODEL_PATH=./models/bert_classifier
```

---

## 📦 Dependencies

See `requirements.txt`. Key ones include:
- `transformers`
- `torch`
- `fastapi`
- `spacy`
- `scikit-learn`
- `streamlit`

---

## 🧾 License

MIT License

---

## 👩‍💻 Author

Developed by anahita. Built for real-world classification and privacy-safe email processing.
