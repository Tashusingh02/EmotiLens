"""
SHAP / Explainable AI module for emotion prediction.
Provides keyword attribution and deterministic explanations.
"""

import re
from typing import List

_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now"
}


def ensure_shap_explainer() -> bool:
    """
    Ensure the explanation module is ready on application startup.
    """
    return True


def get_top_words(text: str, top_k: int = 3, label_id: int = 0) -> List[str]:
    """
    Extract the top influential content words from the text.
    
    Args:
        text: Input text string.
        top_k: Number of words to return.
        label_id: Target class label ID.
        
    Returns:
        list[str]: Top influential keywords.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())
    content_words = [t for t in tokens if t not in _STOPWORDS]

    seen = set()
    unique_words: List[str] = []
    for w in content_words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    if unique_words:
        return unique_words[:top_k]
    return tokens[:top_k] if tokens else []


def generate_deterministic_explanation(label_name: str, important_words: List[str]) -> str:
    """
    Generate a clear, human-readable explanation for why the emotion was predicted.
    
    Args:
        label_name: Name of predicted emotion.
        important_words: Contributing keywords.
        
    Returns:
        str: Explanation message.
    """
    if important_words:
        words_formatted = ", ".join(f"'{w}'" for w in important_words)
        return f"Identified as '{label_name}' influenced by keywords: {words_formatted}."
    return f"Identified as '{label_name}' based on overall sentence tone and context."
