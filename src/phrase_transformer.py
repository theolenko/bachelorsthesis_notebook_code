from __future__ import annotations
from typing import Iterable, List, Optional
from gensim.models.phrases import Phrases, Phraser
from sklearn.base import BaseEstimator, TransformerMixin


class PhraseDetector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for gensim Phrases/Phraser.
    
    This transformer detects and combines common phrases in text data using
    statistical co-occurrence patterns. It's designed to be used in ML pipelines
    to automatically identify bigrams and trigrams that frequently appear together.
    
    Input: List of pre-cleaned strings (without phrase detection applied).
    Output: List of strings with detected phrases connected by delimiter (default: '_').
    
    Benefits:
    - No feature leakage: fit() only sees training split/fold data.
    - HPO-compatible: min_count, threshold, and enable parameters can be optimized.
    - Seamless integration with sklearn pipelines and cross-validation.
    
    Example:
        >>> detector = PhraseDetector(min_count=5, threshold=100.0)
        >>> texts = ["machine learning", "machine learning algorithm", "deep learning"]
        >>> detector.fit(texts)
        >>> detector.transform(["machine learning is great"])
        ["machine_learning is great"]
    """

    def __init__(
        self,
        enable: bool = True,
        min_count: int = 10,
        threshold: float = 300.0,
        delimiter: str = "_",
    ):
        """
        Initialize the PhraseDetector.
        
        Args:
            enable (bool): Whether to enable phrase detection. If False, acts as pass-through.
            min_count (int): Minimum number of times a phrase must appear to be considered.
            threshold (float): Threshold for phrase formation. Higher values = more conservative.
            delimiter (str): Character used to connect words in detected phrases.
        """
        self.enable = enable
        self.min_count = min_count
        self.threshold = threshold
        self.delimiter = delimiter
        self.phraser_: Optional[Phraser] = None

    def fit(self, X: Iterable[str], y=None):
        """
        Learn phrase patterns from training data.
        
        Args:
            X (Iterable[str]): Training texts to learn phrases from.
            y: Ignored, present for sklearn compatibility.
            
        Returns:
            self: Returns the fitted transformer.
        """
        if not self.enable:
            # Skip learning if disabled, only pass-through
            return self
        tokenized: List[List[str]] = [str(x).split() for x in X]
        phrases = Phrases(
            tokenized,
            min_count=self.min_count,
            threshold=self.threshold,
            delimiter=self.delimiter,
        )
        self.phraser_ = Phraser(phrases)
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        """
        Apply learned phrase detection to new texts.
        
        Args:
            X (Iterable[str]): Texts to transform with phrase detection.
            
        Returns:
            List[str]: Transformed texts with detected phrases connected by delimiter.
        """
        if not self.enable or self.phraser_ is None:
            # Pass-through if disabled or not fitted
            return list(X)
        tokenized: List[List[str]] = [str(x).split() for x in X]
        return [" ".join(self.phraser_[toks]) for toks in tokenized]
