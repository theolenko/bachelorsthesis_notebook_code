import re
import unicodedata

def clean_text(text: str, mode: str = "basic") -> str:
    """
    Main function for text cleaning.
    Currently available: mode = "basic"

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
    else:
        raise ValueError("Mode not supported: Use 'basic'.")

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

    # 8a. Remove runs of symbols (dot, dash, underscore)
    # Any sequence of 3 or more consecutive '.', '-', or '_' characters—regardless of order or mix—will be replaced by a single space.
    # Example: 'hello---world' -> 'hello world', 'foo___bar' -> 'foo bar', 'wait...' -> 'wait '
    #          'foo-._--__...bar' -> 'foo bar'
    text = re.sub(r"[\.\-\_]{3,}", " ", text)
    # 8b. Remove most punctuation and typographic symbols
    # Example: 'hello, world!' -> 'hello  world '
    #          '«quote»' -> ' quote '
    text = re.sub(r"[!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~]", " ", text)
    text = re.sub(r"[„“‚‘‹›«»\"]", " ", text)

    # 9. Remove control characters (ASCII control chars and replacement char)
    # Example: 'foo\x00bar' -> 'foobar'
    text = re.sub(r"[\u0000-\u001F\u007F\uFFFD]", "", text)

    # 10. Normalize whitespace (collapse multiple spaces, trim)
    # Example: '  hello   world  ' -> 'hello world'
    text = re.sub(r"\s+", " ", text).strip()

    return text