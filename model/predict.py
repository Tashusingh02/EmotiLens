"""
Emotion prediction module used by FastAPI app and CLI scripts.
Bridges to the Hugging Face GoEmotions sequence classification model.
"""

import torch
from hybrid.emotion_detector import _load_model


def load_model(model_path=None):
    """
    Ensure the emotion detection model and tokenizer are loaded into memory.
    """
    return _load_model(model_path)


def predict_emotion(text: str) -> tuple[int, str, float]:
    """
    Predict emotion for a given input text.
    
    Args:
        text: Input string to classify.
        
    Returns:
        tuple[int, str, float]: (label_id, label_name, probability)
    """
    model, tokenizer, label_names = _load_model()

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    predicted_id = int(torch.argmax(probabilities, dim=-1).item())
    probability = float(probabilities[0][predicted_id].item())
    label_name = label_names[predicted_id] if predicted_id < len(label_names) else f"LABEL_{predicted_id}"

    return predicted_id, label_name, probability
