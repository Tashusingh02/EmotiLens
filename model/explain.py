"""
Occlusion-based Explainable AI module for emotion prediction.
Calculates word importance by measuring the drop in prediction probability
when each word is individually removed (occluded).
"""

from typing import List, Tuple, Union
import torch
from hybrid.emotion_detector import _load_model

# Minimum probability drop threshold for a word to be considered influential
MIN_DROP = 0.03


def ensure_explainer() -> bool:
    """
    Ensure the emotion classification model and tokenizer are loaded and warmed up.
    """
    _load_model()
    return True


def get_top_words_with_scores(
    text: str,
    top_k: int = 3,
    label_id: int | None = None
) -> List[Tuple[str, float]]:
    """
    Extract the top influential words and their importance scores using occlusion analysis.
    
    Splits text into words (max 40), creates variants with each word removed,
    and runs them through the emotion model in a single batched forward pass.
    The importance score is the drop in predicted emotion probability:
        score = P(label | original) - P(label | word removed)
    
    Args:
        text: Input text string.
        top_k: Number of words to return.
        label_id: Target class label ID. If None, uses the model's top predicted label.
        
    Returns:
        List of (word, score) tuples for the words with the largest positive drops.
    """
    raw_words = (text or "").split()[:40]
    if not raw_words:
        return []

    model, tokenizer, label_names = _load_model()
    device = next(model.parameters()).device

    # Build original text + one variant per word with that word removed
    variants = [text]
    for i in range(len(raw_words)):
        remaining = raw_words[:i] + raw_words[i + 1:]
        variant_text = " ".join(remaining) if remaining else " "
        variants.append(variant_text)

    # Batched forward pass with dynamic padding for speed
    inputs = tokenizer(
        variants,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    # Determine target label if not provided
    if label_id is None or label_id < 0 or label_id >= probs.shape[-1]:
        label_id = int(torch.argmax(probs[0]).item())

    orig_prob = float(probs[0, label_id].item())

    # Score each word by the drop in target label probability
    seen = set()
    scored = []
    for i, w in enumerate(raw_words):
        clean_w = w.strip(".,!?:;\"'()[]{}").lower()
        if not clean_w:
            clean_w = w.lower()
        if clean_w in seen:
            continue
        seen.add(clean_w)

        occ_prob = float(probs[i + 1, label_id].item())
        drop = orig_prob - occ_prob
        scored.append((clean_w, drop))

    # Only return words whose occlusion drop is at least MIN_DROP (0.03)
    qualifying = [item for item in scored if item[1] >= MIN_DROP]
    qualifying.sort(key=lambda x: x[1], reverse=True)
    return qualifying[:top_k]


def get_top_words(
    text: str,
    top_k: int = 3,
    label_id: int | None = None
) -> List[str]:
    """
    Return the top influential words as a list of strings.
    """
    scored = get_top_words_with_scores(text, top_k=top_k, label_id=label_id)
    return [w for w, _ in scored]


def generate_deterministic_explanation(
    label_name: str,
    important_words: Union[List[Tuple[str, float]], List[str]]
) -> str:
    """
    Generate a clear, human-readable explanation using occlusion analysis.
    
    Args:
        label_name: Name of predicted emotion.
        important_words: Contributing keywords, either (word, score) tuples or word strings.
        
    Returns:
        str: Explanation message.
    """
    if not important_words:
        return "No single word dominated this prediction."

    if isinstance(important_words[0], tuple):
        words_formatted = ", ".join(f"'{w}' (+{score:.3f})" for w, score in important_words)
    else:
        words_formatted = ", ".join(f"'{w}'" for w in important_words)

    return f"Identified as '{label_name}'. Words that most affected this prediction (occlusion analysis): {words_formatted}."
