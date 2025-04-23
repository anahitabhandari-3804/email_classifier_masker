from src.utils import PIIMasker

from src.model import EmailClassifier

class EmailProcessingPipeline:
    def __init__(self, model_path: str):
        self.masker = PIIMasker()
        self.classifier = EmailClassifier()
        self.classifier.load_model(model_path)

    def process_email(self, email: str) -> str:
        masked_email = self.masker.mask_pii(email)
        prediction = self.classifier.predict([masked_email])[0]
        return prediction
