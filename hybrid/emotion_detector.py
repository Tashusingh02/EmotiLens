"""
Emotion detector module that loads the GoEmotions model
and provides functionality to detect emotions from text.
Supports loading a public pretrained model from HuggingFace
("SamLowe/roberta-base-go_emotions") or local weights if available.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

DEFAULT_PRETRAINED_MODEL = "SamLowe/roberta-base-go_emotions"

# Global variables to store loaded model, tokenizer, and label names
_model = None
_tokenizer = None
_label_names = None

def _load_model(model_path=None):
    """
    Load the emotion classification model and tokenizer.
    If local trained weights exist at model_path or './model', load them;
    otherwise load the public pretrained GoEmotions model from HuggingFace.
    
    Args:
        model_path: Optional path to directory containing saved model and tokenizer
    
    Returns:
        tuple: (model, tokenizer, label_names)
    """
    global _model, _tokenizer, _label_names
    
    if _model is None or _tokenizer is None:
        # Check if a valid local model directory exists
        if model_path and os.path.isdir(model_path) and any(os.scandir(model_path)):
            target = model_path
        elif os.path.isdir("./model") and any(os.scandir("./model")):
            files = [f.name for f in os.scandir("./model")]
            if any(f.endswith((".bin", ".safetensors", ".pt")) or f == "config.json" for f in files):
                target = "./model"
            else:
                target = DEFAULT_PRETRAINED_MODEL
        else:
            target = DEFAULT_PRETRAINED_MODEL
        
        try:
            _tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
            _model = AutoModelForSequenceClassification.from_pretrained(target, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(target)
            _model = AutoModelForSequenceClassification.from_pretrained(target)
        _model.eval()  # Set model to evaluation mode
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(device)
    
    # Load label names from the model config id2label mapping
    if _label_names is None:
        if hasattr(_model.config, "id2label") and _model.config.id2label:
            _label_names = [_model.config.id2label[i] for i in range(len(_model.config.id2label))]
        else:
            _label_names = [
                "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"
            ]
    
    return _model, _tokenizer, _label_names

def detect_emotion(text):
    """
    Detect the emotion label for the given text.
    
    Args:
        text: Input text string to classify
    
    Returns:
        str: The emotion label name (e.g., "joy", "sadness", "anger")
    """
    # Load model if not already loaded
    model, tokenizer, label_names = _load_model()
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get the predicted label ID
    predicted_id = torch.argmax(probabilities, dim=-1).item()
    
    # Get the label name
    label_name = label_names[predicted_id] if predicted_id < len(label_names) else f"LABEL_{predicted_id}"
    
    return label_name






