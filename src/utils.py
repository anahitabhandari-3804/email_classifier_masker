import spacy
import re
import pandas as pd
import pickle
import os
from typing import Tuple, List

class PIIMasker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.phone_pattern = r"\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b"
        self.mask_token = "[MASKED]"

    def mask_pii(self, text: str) -> str:
        # Mask phone numbers using regex
        masked_text = re.sub(self.phone_pattern, self.mask_token, text)
        
        # Mask entities using spaCy NER
        doc = self.nlp(masked_text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG", "EMAIL"]:
                masked_text = masked_text.replace(ent.text, self.mask_token)
        return masked_text

    def preprocess_data(self, input_path: str, output_path: str, original_path: str) -> None:
        # Read dataset
        df = pd.read_csv("combined_emails_with_natural_pii.csv")
        
        # Store original data securely
        with open(original_path, 'wb') as f:
            pickle.dump(df, f)
        
        # Apply PII masking
        df['masked_email'] = df['email'].apply(self.mask_pii)
        
        # Save masked data
        df[['masked_email', 'type']].to_csv(output_path, index=False)

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def load_original_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as f:
        return pickle.load(f)