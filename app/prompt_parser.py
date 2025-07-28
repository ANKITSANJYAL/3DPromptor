import spacy
from typing import List, Tuple

# Load spaCy English model (should be installed via requirements.txt)
nlp = spacy.load("en_core_web_sm")

# Style cue keywords (extend as needed)
STYLE_KEYWORDS = {
    "style", "painting", "render", "art", "shader", "design", "texture", "look",
    "aesthetic", "vibe", "fashion", "graffiti", "cel-shaded", "cubist",
    "pixel", "cyberpunk", "low-poly", "clay", "lego", "watercolor",
    "ukiyo-e", "ink", "comic", "dreamlike", "oil", "sketch"
}

def split_prompt_tokens(prompt: str) -> Tuple[List[str], List[str]]:
    """
    Splits the prompt into object tokens (Tobj) and style tokens (Tstyle).

    Args:
        prompt: Natural language prompt

    Returns:
        (Tobj, Tstyle): Two lists of content and style words
    """
    doc = nlp(prompt.lower())
    object_tokens = set()
    style_tokens = set()

    for token in doc:
        # Noun chunks = object tokens
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            object_tokens.add(token.text)

        # Adjectives or style-related nouns
        if token.pos_ == "ADJ" or token.lemma_ in STYLE_KEYWORDS:
            style_tokens.add(token.text)

        # Catch "in [style] style" patterns
        if token.text in STYLE_KEYWORDS:
            style_tokens.add(token.text)

    return list(object_tokens), list(style_tokens)
