import re
from gensim.models.phrases import Phrases, Phraser
from src.cleaning import _clean_basic


def train_phrases(
    raw_texts,
    min_count: int = 5,
    threshold: float = 10.0,
    delimiter: str = '_'
) -> Phraser:
    """
    Train and return a Gensim Phraser for multiword phrase detection.

    Args:
        raw_texts (List[str]): List of raw document strings.
        min_count (int): Minimum count threshold for phrase candidates.
        threshold (float): PMI-based score threshold for forming phrases.
        delimiter (bytes): Delimiter for joined phrases (e.g., b'_').

    Returns:
        Phraser: Trained Gensim Phraser instance.
    """
    # Prepare tokenized corpus with basic cleaning
    docs = []
    for text in raw_texts:
        cleaned_basic = _clean_basic(text) #clean the text with basic cleaning function
        # Remove non-alphanumeric characters and split into tokens
        cleaned = re.sub(r'[^a-zäöüß\s]', ' ', cleaned_basic)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = cleaned.split()
        # remove single letter tokens
        tokens = [t for t in tokens if len(t) > 1]
        docs.append(tokens)

    # Train Phrases model and create fast Phraser
    phrases = Phrases(docs, min_count=min_count, threshold=threshold, delimiter=delimiter)
    bigram = Phraser(phrases)
    return bigram
