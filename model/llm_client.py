"""
LLM Client module for EmotiLens support and book recommendations.
Reuses Ollama chat integration from hybrid.local_llm with graceful fallbacks
when Ollama or network services are unavailable.
"""

import os
import re
import requests
from typing import List, Optional, Tuple, Any

# Re-use Ollama integration and style prompts from hybrid package
from hybrid.local_llm import chat, check_ollama_connection
from hybrid.style_prompts import get_system_prompt, format_user_prompt, get_emotion_bucket

# Curated fallback responses for each emotion bucket when Ollama is unavailable
FALLBACK_RESPONSES = {
    "sadness": (
        "I can hear how difficult things feel right now. It is completely valid to feel sad, "
        "and you don't have to carry it all alone. Take things one gentle moment at a time—I'm here to listen."
    ),
    "fear": (
        "It is completely understandable to feel anxious or nervous right now. Take a slow, steady breath. "
        "Whatever you're facing, we can take it one small step at a time."
    ),
    "anger": (
        "It sounds like you're dealing with something really frustrating. Your feelings are valid, and it's okay "
        "to feel angry. Take a pause to breathe, and feel free to vent or talk through it with me."
    ),
    "joy": (
        "That is wonderful to hear! Positive moments like this are worth cherishing—take time to celebrate "
        "and savor how you're feeling right now."
    ),
    "neutral": (
        "Thank you for sharing your thoughts with me. I'm here to support you—how are you doing, and what's on your mind?"
    )
}

# Curated fallback book recommendations by emotion bucket when Google Books API is unreachable
FALLBACK_BOOKS = {
    "sadness": [
        (
            "The Upward Spiral: Using Neuroscience to Reverse the Course of Depression",
            "Neuroscience-based strategies to reshape brain circuits and find relief from depression."
        ),
        (
            "Feeling Good: The New Mood Therapy",
            "Clinically proven cognitive behavioral therapy techniques to overcome depression and low mood."
        ),
        (
            "Reasons to Stay Alive",
            "A heartfelt, uplifting memoir of overcoming depression and reclaiming life."
        ),
    ],
    "fear": [
        (
            "The Anxiety and Phobia Workbook",
            "Practical cognitive-behavioral tools and relaxation techniques to overcome anxiety and panic."
        ),
        (
            "Dare: The New Way to End Anxiety and Stop Panic Attacks",
            "A fast, natural approach to overcoming anxiety, panic attacks, and worry."
        ),
        (
            "First, We Make the Beast Beautiful",
            "A compassionate and practical exploration of living with and understanding anxiety."
        ),
    ],
    "anger": [
        (
            "Anger: Wisdom for Cooling the Flames",
            "Mindfulness practices by Thich Nhat Hanh for transforming anger into compassion."
        ),
        (
            "The Dance of Anger",
            "A classic, empowering guide to transforming anger into a constructive force in personal relationships."
        ),
        (
            "Rage: A Step-by-Step Guide to Overcoming Explosive Anger",
            "Clear cognitive strategies to recognize emotional triggers and manage intense frustration."
        ),
    ],
    "joy": [
        (
            "The How of Happiness",
            "A scientific approach to understanding happiness and building sustainable emotional flourishing."
        ),
        (
            "The Book of Joy",
            "Enduring insights on lasting happiness and compassion shared by the Dalai Lama and Desmond Tutu."
        ),
        (
            "Flourish: A Visionary New Understanding of Happiness and Well-being",
            "A groundbreaking work on positive psychology and living a fulfilling life."
        ),
    ],
    "neutral": [
        (
            "Atomic Habits",
            "A proven framework for improving every day through small, consistent behavioral habits."
        ),
        (
            "Wherever You Go, There You Are",
            "A gentle, grounding introduction to daily mindfulness and present-moment awareness."
        ),
        (
            "Mindset: The New Psychology of Success",
            "How our conscious and unconscious mindset affects personal growth and resilience."
        ),
    ],
}


def _normalize_title(title: str) -> str:
    """Normalize book title for deduplication comparison."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").strip().lower())


def generate_support_response(
    text: str,
    label_name: str,
    advice_given: bool = False,
    history: Optional[List[Any]] = None,
) -> str:
    """
    Generate an empathetic support response using local Ollama (qwen2.5:1.5b),
    using a concise system prompt and falling back gracefully if Ollama is not running.
    
    Args:
        text: User's latest input text.
        label_name: Detected emotion name (e.g. 'joy', 'sadness', 'anger').
        advice_given: Whether coping advice has already been shared.
        history: Prior conversation turns (list of objects or dicts with role and content).
        
    Returns:
        str: Conversational assistant response.
    """
    bucket = get_emotion_bucket(label_name)

    # Concise ~60-word system prompt tailored to the detected emotion
    system_prompt = (
        f"You are EmotiLens, a compassionate mental health companion. "
        f"The user is feeling {label_name}. "
        f"Reply warmly, empathetically, and constructively in 2 to 3 short sentences. "
        f"Acknowledge their emotional state with genuine care, keep your tone supportive and human, "
        f"and gently encourage them without being overly clinical or repetitive."
    )
    if advice_given:
        system_prompt += " Keep focus on gentle listening rather than repeating advice."

    # Format messages for the Ollama chat interface
    messages = [{"role": "system", "content": system_prompt}]

    # Send only the last 2 chat turns as history to keep inference fast
    recent_history = history[-2:] if history else []
    for item in recent_history:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if role and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": text})

    # Attempt to query Ollama via hybrid.local_llm with speed optimizations
    try:
        response = chat(
            messages,
            model="qwen2.5:1.5b",
            keep_alive="30m",
            options={
                "num_predict": 100,
                "temperature": 0.6,
                "num_ctx": 1024,
            }
        )
        if response and response.strip():
            return response.strip()
    except Exception:
        # Graceful fallback when Ollama is offline or unavailable
        pass

    return FALLBACK_RESPONSES.get(bucket, FALLBACK_RESPONSES["neutral"])


def suggest_single_book(
    emotion: str,
    important_words: Optional[List[str]] = None,
    excluded_titles: Optional[List[str]] = None,
) -> Optional[Tuple[str, str]]:
    """
    Suggest a single relevant book title and concise description.
    Queries the Google Books API if GOOGLE_BOOKS_API_KEY is configured and accessible,
    or falls back to a curated library of safe mental-health book recommendations.
    
    Args:
        emotion: The target emotion string.
        important_words: Contextual words identified from the user's input.
        excluded_titles: Titles already presented to the user to avoid repetition.
        
    Returns:
        tuple[str, str]: (book_title, book_description) or None if unavailable.
    """
    excluded_norm = {_normalize_title(t) for t in (excluded_titles or []) if t}
    safe_emotion = (emotion or "").strip().lower() or "mental health"
    bucket = get_emotion_bucket(safe_emotion)

    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")

    # 1. Try Google Books API lookup if key is configured
    if api_key and api_key.strip():
        keywords = [w.strip().lower() for w in (important_words or []) if isinstance(w, str) and w.strip()]
        top_keywords = " ".join(keywords[:2])
        search_query = f"self help {safe_emotion} {top_keywords}".strip()

        try:
            url = "https://www.googleapis.com/books/v1/volumes"
            resp = requests.get(
                url,
                params={"q": search_query, "maxResults": 10, "key": api_key.strip()},
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items") or []
                for item in items:
                    volume_info = item.get("volumeInfo") or {}
                    title = volume_info.get("title")
                    if not title or _normalize_title(title) in excluded_norm:
                        continue
                    desc = volume_info.get("description") or f"A helpful self-help book for dealing with {safe_emotion}."
                    # Keep description concise
                    short_desc = desc[:300].rsplit(".", 1)[0] + "." if len(desc) > 300 and "." in desc[:300] else desc[:300]
                    return str(title), short_desc
        except Exception:
            # Fall through to curated fallback library if network or API fails
            pass

    # 2. Curated fallback library based on emotion bucket
    candidates = FALLBACK_BOOKS.get(bucket, FALLBACK_BOOKS["neutral"])
    for title, desc in candidates:
        if _normalize_title(title) not in excluded_norm:
            return title, desc

    # 3. If all bucket-specific candidates were excluded, try neutral shelf
    for title, desc in FALLBACK_BOOKS["neutral"]:
        if _normalize_title(title) not in excluded_norm:
            return title, desc

    # Return first available book if all were seen
    if candidates:
        return candidates[0]
    return FALLBACK_BOOKS["neutral"][0]
