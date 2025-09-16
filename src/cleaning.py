import re
import os
import unicodedata
import nltk
#load stopwords more robustly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer



# Load German stopwords and stemmer for advanced cleaning/preprocessing
_german_stop = set(stopwords.words("german"))
_stemmer = SnowballStemmer("german")


def clean_text(text: str, mode: str = "basic") -> str:
    """
    Main function for text cleaning.
    Currently available: mode = "basic", "advanced"

    Args:
        text (str): The input text to be cleaned.
        mode (str, optional): Cleaning mode. Only 'basic' is supported currently.

    Returns:
        str: The cleaned text.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    if mode == "basic":
        return _clean_basic(text)
    elif mode == "advanced":
        return _clean_advanced(text)
    else:
        raise ValueError("Mode not supported: Use 'basic' or 'advanced'.")
    

def _clean_basic(text: str) -> str:
    """
    Basic text cleaning pipeline. Applies a series of normalization and cleaning steps to the input text.

    Steps:
        1. Unicode normalization (NFKC form)
        2. Remove escape sequences (e.g., \n, \t, \r, escaped quotes)
        3. Convert to lowercase
        4. Reconstruct separated letter sequences (e.g., 's t r u c t u r e' -> 'structure')
        5. Remove hyphens/underscores in compound words and at line breaks
        6. Remove single-letter words
        7. Remove blocks of tokens dominated by numbers
        8. Remove symbol runs and typographic characters
        9. Remove control characters
       10. Normalize whitespace

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # 1. Unicode normalization (NFKC form)
    # Example: "e01" (e + combining accent) -> "é"
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove escape sequences and escaped quotes
    # Example: 'He said: \\"Hello\\"' -> 'He said: "Hello"'
    #          'Line1\nLine2' -> 'Line1 Line2'
    text = text.replace('\"', '"').replace("\\'", "'")
    text = re.sub(r"\\[ntr]", " ", text)  # Replace \n, \t, \r with space

    # 3. Convert to lowercase
    # Example: 'Hello WORLD' -> 'hello world'
    text = text.lower()

    # 3b. Remove emojis
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map
        u"\U0001F1E0-\U0001F1FF"  # Flags
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # 4. Reconstruct separated letter sequences
    # Example: 's t r u c t u r e' -> 'structure'
    #          'b a s i c   t e x t' -> 'basic text'
    text = re.sub(r"\b(?:[a-zäöüß]\s){2,}[a-zäöüß]\b", lambda m: m.group(0).replace(" ", ""), text)

    # 5a. Remove hyphens/underscores in the middle of words
    # Example: 'struktur-_wandel' -> 'strukturwandel'
    text = re.sub(r"(\w)[-_](\w)", r"\1\2", text)
    # 5b. Remove isolated hyphens/underscores between words
    # Example: 'foo - bar' -> 'foo bar', 'foo _ bar' -> 'foo bar'
    text = re.sub(r"\s[-_]\s", " ", text)
    # 5c. Remove hyphens at line breaks
    # Example: 'staats- regierung' -> 'staatsregierung'
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # 6. Remove single-letter words
    # Example: 'a quick b brown c fox' -> 'quick brown fox'
    text = re.sub(r"\b[a-zäöüß]\b", "", text)

    # 7. Remove blocks of tokens dominated by numbers
    # Example: 'foo 123 456 789 101 112 131 415 161 718 192 021 222 324 252 627 bar' (if >50% numbers in 15 tokens, block is removed)
    tokens = text.split()
    cleaned_tokens = []
    buffer = []
    for token in tokens:
        buffer.append(token)
        # Process buffer every 15 tokens
        if len(buffer) >= 15:
            # Calculate ratio of numeric tokens in buffer
            # Example: buffer = ['foo', '123', '456', ...] -> numeric_ratio = 14/15
            numeric_ratio = sum(t.replace(".", "").replace(",", "").isdigit() for t in buffer) / len(buffer)
            if numeric_ratio > 0.5:
                # If more than 50% are numbers, discard buffer
                # Example: buffer = ['123', '456', ..., '789'] -> buffer is cleared
                buffer = []
            else:
                # Otherwise, keep buffer
                cleaned_tokens.extend(buffer)
                buffer = []
    # Add any remaining tokens
    cleaned_tokens.extend(buffer)
    text = " ".join(cleaned_tokens)

    # 8. Remove runs of symbols (dot, dash, underscore)
    # Any sequence of 3 or more consecutive '.', '-', or '_' characters—regardless of order or mix—will be replaced by a single space.
    # Example: 'hello---world' -> 'hello world', 'foo___bar' -> 'foo bar', 'wait...' -> 'wait '
    #          'foo-._--__...bar' -> 'foo bar'
    text = re.sub(r"[\.\-\_]{3,}", " ", text)

    # 9. Remove control characters (ASCII control chars and replacement char)
    # Example: 'foo\x00bar' -> 'foobar'
    text = re.sub(r"[\u0000-\u001F\u007F\uFFFD]", "", text)

    # 10. Normalize whitespace (collapse multiple spaces, trim)
    # Example: '  hello   world  ' -> 'hello world'
    text = re.sub(r"\s+", " ", text).strip()

    return text

def _clean_advanced(text: str) -> str:
    """
    Advanced text cleaning pipeline for German federal parliamentary documents.
    Builds on basic cleaning and adds domain-specific processing, multiword grouping, 
    stopword removal, and stemming.
    
    Steps:
        1. Apply basic cleaning pipeline
        2. Remove URLs and email addresses  
        3. Remove parliamentary boilerplate patterns
        4. Remove non-alphabetic characters
        5. Normalize whitespace
        6. Initial tokenization and single character filtering
        7. Multiword grouping via trained Phraser (not done here, but for each split during model training in extra module)
        8. Stopword removal
        9. Phrase-aware stemming
       10. Final whitespace normalization
    """
    # 1. Start from basic cleanup
    # Apply all basic normalization steps (unicode, lowercase, symbol removal, etc.)
    text = _clean_basic(text)
    
    # 2. Remove URLs and email addresses
    # Example: 'Visit https://example.com or email test@domain.de' -> 'Visit  or email '
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', ' ', text)
    
    # 3. Remove parliamentary boilerplate patterns
    # Example: 'Text drucksache 7/1234 mehr text' -> 'Text  mehr text'
    text = re.sub(r'drucksache\s+\d+/\d+[^\w]*', ' ', text)
    
    # Remove common parliamentary boilerplate phrases (case-insensitive matching)
    # Example: 'der sächsische landtag beschließt' -> ' beschließt'
    # Example: 'mit freundlichen grüßen dr müller' -> ' dr müller'
    for p in ["landtag", "hausanschrift", "mit freundlichen grüßen", "wahlperiode",
              "namens und im auftrag", "anlage", "seite", "aktenzeichen", "acrobat reader",
              "verarbeitung personenbezogener daten", "amtsbekannter rex"]:
        # Match the phrase with word boundaries to avoid partial matches
        text = re.sub(rf'\b{re.escape(p)}\b', ' ', text)
    
    # 4. Remove non-alphabetic characters (keep spaces) - BEFORE tokenization
    # This matches the preprocessing used during phraser training, and is automatically done during tfidf vectorizaion with TFIDFVectorizer
    # Example: 'klimaschutz-maßnahmen 2024!' -> 'klimaschutz maßnahmen '
    text = re.sub(r'[^a-zäöüß\s]', ' ', text)
    
    # 5. Normalize whitespace and tokenize
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    
    # 6. Remove single character tokens (matching phraser training preprocessing)
    # Example: ['klimaschutz', 'maßnahmen', 'a', 'für'] -> ['klimaschutz', 'maßnahmen', 'für']
    tokens = [t for t in tokens if len(t) > 1]
    
    # 7. Multiword grouping via trained Phraser (if available)
    # The phraser detects common word combinations and joins them with underscores
    # Example: ['soziale', 'gerechtigkeit'] -> ['soziale_gerechtigkeit']
    # Example: ['klimaschutz', 'maßnahmen'] -> ['klimaschutz_maßnahmen'] 
    # IMPORTANT: Text preprocessing here matches exactly the preprocessing used during training
    # done seperately during model training on dev data to prevent data leakage
    
    # 7. Stopword removal and additional filtering
    # Example: ['die', 'regierung_will', 'neue', 'gesetze'] -> ['regierung_will', 'neue', 'gesetze']
    # Removes German stopwords (der, die, das, und, etc.)
    tokens = [t for t in tokens if t not in _german_stop and len(t) > 1]
    
    # 9. Stemming - reduce words to their root form with phrase-aware processing
    # For single words: ['regierung', 'gesetze'] -> ['regier', 'gesetz']
    # For multiword phrases: ['soziale_gerechtigkeit'] -> ['sozial_gerecht']
    # Uses German Snowball stemmer while preserving phrase structure
    tokens = [_stemmer.stem(t) for t in tokens]
    
    # 10. Final whitespace normalization
    # Ensure single spaces between tokens and trim
    result = " ".join(tokens)
    return re.sub(r'\s+', ' ', result).strip()

