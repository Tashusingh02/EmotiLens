"""
Crisis and self-harm detection module.
Provides detection logic and official helpline responses.
"""

import re

CRISIS_DIRECT_PATTERNS = [
    re.compile(r"\b(?:kill|killing|end|ending|harm|harming|hurt|hurting|cut|cutting|hang|hanging|slit)\s+myself\b", re.I),
    re.compile(r"\b(?:want|wanna|wish)\s+(?:to\s+die|i\s+(?:was|were)\s+dead)\b", re.I),
    re.compile(r"\b(?:better\s+off\s+dead|better\s+off\s+without\s+me)\b", re.I),
    re.compile(r"\bsuicid(?:e|al)\b", re.I),
    re.compile(r"\b(?:take|taking)\s+my\s+(?:own\s+)?life\b", re.I),
    re.compile(r"\b(?:end|ending)\s+my\s+life\b", re.I),
    re.compile(r"\bself[- ]?harm\b", re.I),
    re.compile(r"\bslit\s+my\s+(?:wrist|wrists|throat)\b", re.I),
    re.compile(r"\b(?:intentional\s+)?overdose\b", re.I),
    re.compile(r"\bend\s+it\s+all\b", re.I),
    re.compile(r"\b(?:don'?t|dont)\s+want\s+to\s+live\b", re.I),
    re.compile(r"\bno\s+reason\s+to\s+live\b", re.I),
    re.compile(r"\bnot\s+worth\s+living\b", re.I),
    re.compile(r"\btired\s+of\s+living\b", re.I),
    re.compile(r"\bdone\s+with\s+(?:my\s+)?life\b", re.I),
]

CRISIS_CONTEXTUAL_PATTERNS = [
    re.compile(r"\bcan'?t\s+(?:take|go\s+on|handle)\s+(?:this|it)\s+anymore\b", re.I),
    re.compile(r"\bgiv(?:e|ing)\s+up\s+on\s+life\b", re.I),
    re.compile(r"\bnothing\s+matters\s+anymore\b", re.I),
    re.compile(r"\bdisappear\s+forever\b", re.I),
    re.compile(r"\beveryone\s+would\s+be\s+better\s+off\b", re.I),
    re.compile(r"\bno\s+way\s+out\b", re.I),
]

CRISIS_DISTRESS_EMOTIONS = {
    "sadness", "grief", "remorse", "fear", "nervousness", "disappointment", "anger", "disapproval"
}

CRISIS_SUPPORT_RESPONSE = (
    "It sounds like you are carrying an overwhelming amount of pain right now, "
    "and I want you to know that you are not alone and your life matters. "
    "Please connect immediately with trained professionals who are ready to support you with free, "
    "confidential help:\n\n"
    "• Tele-MANAS: Call 14416 (or 1800-891-4416) — 24/7 National Mental Health Helpline\n"
    "• iCall: Call 9152987821 — Mon-Sat, 8 AM to 10 PM Psychosocial Helpline\n"
    "• Emergency Services: Call 112\n\n"
    "Please reach out to one of these resources or someone you trust right now. People who care are waiting to listen and support you."
)


def detect_crisis(text: str, emotion: str = "") -> bool:
    """
    Check if the user input indicates crisis, self-harm, or suicidal ideation.
    Evaluates direct intent keywords as well as contextual distress phrases
    combined with high-risk emotional signals.
    """
    raw_text = (text or "").strip().lower()
    if not raw_text:
        return False

    # 1. Direct crisis / self-harm keyword or phrase match
    for pattern in CRISIS_DIRECT_PATTERNS:
        if pattern.search(raw_text):
            return True

    # 2. Contextual distress phrase combined with a distress emotion signal
    safe_emotion = (emotion or "").strip().lower()
    if safe_emotion in CRISIS_DISTRESS_EMOTIONS:
        for pattern in CRISIS_CONTEXTUAL_PATTERNS:
            if pattern.search(raw_text):
                return True

    return False
