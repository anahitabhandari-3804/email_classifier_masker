from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from src.pipeline import EmailProcessingPipeline
import spacy

# Initialize SpaCy for PII entity recognition
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Email Support Classifier API")

# Initialize the email processing pipeline (ensure correct path to the trained model)
pipeline = EmailProcessingPipeline(model_path="C:/Users/DELL/Desktop/xyz/akaike/EmailSupportClassifier/models/bert_classifier")
# Define the request and response models
from pydantic import BaseModel
from typing import List, Tuple
from typing import Optional
from typing import Dict, Any
from typing import Union
from typing import Callable
from typing import Type
from typing import Any, Dict, List, Optional, Tuple
from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute


class EmailRequest(BaseModel):
    email_text: str

class EntityInfo(BaseModel):
    position: Tuple[int, int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[EntityInfo]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    try:
        raw_text = request.email_text
        doc = nlp(raw_text)
        entities = []

        # Identify PII entities using SpaCy
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "EMAIL"]:
                start = ent.start_char
                end = ent.end_char
                entities.append({
                    "position": [start, end],
                    "classification": ent.label_,
                    "entity": ent.text
                })

        # Mask entities in the email
        masked_text = raw_text
        for ent in sorted(entities, key=lambda e: e["position"][0], reverse=True):
            start, end = ent["position"]
            masked_text = masked_text[:start] + "[MASKED]" + masked_text[end:]

        # Classify the email using the trained model
        category = pipeline.process_email(raw_text)  # Ensure this method is correct in your pipeline

        return {
            "input_email_body": raw_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_text,
            "category_of_the_email": category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
