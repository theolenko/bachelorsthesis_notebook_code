from gensim.models.phrases import Phrases, Phraser

def fit_phraser_on_texts(texts, min_count=10, threshold=300.0, delimiter="_"):
    """
    texts: Liste von Strings, die bereits 'advanced' bereinigt sind
           (Stopwörter entfernt, gestemmt) — aber noch OHNE Phraser.
    """
    tokenized = [t.split() for t in texts]
    phrases = Phrases(tokenized, min_count=min_count, threshold=threshold, delimiter=delimiter)
    return Phraser(phrases)

def apply_phraser(texts, phraser):
    tokenized = [t.split() for t in texts]
    return [" ".join(phraser[toks]) for toks in tokenized]
