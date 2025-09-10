import os
import tempfile
import numpy as np
import fasttext
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils._tags import Tags
from types import SimpleNamespace

class FastTextSklearnClassifier(BaseEstimator, ClassifierMixin):
    """
    A lightweight scikit-learn compatible wrapper around fastText's
    supervised classifier.

    Key decisions:
      - Default loss='softmax' and thread=1 for deterministic behavior.
      - No oversampling by default (pos_oversample=1.0).
      - Deterministic data order: we do NOT shuffle the training lines.
      - Predict uses batched fastText .predict for speed.
    """
    _estimator_type = "classifier"   # ← Attribut, keine Property!

    def __init__(self,
                 lr=0.1, dim=100, ws=5, epoch=25, minCount=2,
                 minn=2, maxn=4, neg=10, wordNgrams=2,
                 loss="softmax", bucket=2_000_000, thread=1, verbose=0,
                 label_prefix="__label__", pos_oversample=1.0,
                 decision_threshold=0.5):
        # fastText core hyperparameters
        self.lr = lr
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.minCount = minCount
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.wordNgrams = wordNgrams
        self.loss = loss
        self.bucket = bucket
        self.thread = thread
        self.verbose = verbose

        # dataset/label handling
        self.label_prefix = label_prefix
        self.pos_oversample = pos_oversample  # 1.0 == no oversampling
        self.decision_threshold = decision_threshold

        # will hold the trained fastText model
        self.model_ = None

    def __sklearn_is_fitted__(self):
        return getattr(self, "model_", None) is not None

    def __sklearn_tags__(self):
        # scikit-learn 1.7 expects tags via this method
        return SimpleNamespace(
            estimator_type="classifier",
            requires_fit=True,
            no_validation=False,
            X_types=("string,",)
            )

    # internal helpers  
    def _sanitize_text(self, t: str) -> str:
        """Replace newlines/tabs with spaces and strip extra whitespace."""
        if t is None:
            return ""
        # normalize whitespace to avoid accidental line breaks in train.txt
        return " ".join(str(t).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())

    def _to_ft_format(self, X, y) -> str:
        """
        Convert (X, y) to fastText's supervised format:
        '__label__<int> <space> text...'
        We keep the input order (no shuffling) for reproducibility.
        """
        lines = []
        for t, lab in zip(X, y):
            lab = int(lab)
            line = f"{self.label_prefix}{lab} {self._sanitize_text(t)}"
            lines.append(line)
            # Optional positive-class oversampling (disabled by default)
            if lab == 1 and self.pos_oversample > 1.0:
                lines.extend([line] * (int(self.pos_oversample) - 1))
        return "\n".join(lines)

    # sklearn API 

    def fit(self, X, y):
        """
        Train a fastText supervised model.
        The training file is written in a temporary directory and removed after training.
        """
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)

        # fastText reads from disk; write a deterministic, UTF-8 file.
        with tempfile.TemporaryDirectory() as td:
            train_path = os.path.join(td, "train.txt")
            with open(train_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(self._to_ft_format(X, y))

            self.model_ = fasttext.train_supervised(
                input=train_path,
                lr=self.lr,
                dim=self.dim,
                ws=self.ws,
                epoch=self.epoch,
                minCount=self.minCount,
                minn=self.minn,
                maxn=self.maxn,
                neg=self.neg,
                wordNgrams=self.wordNgrams,
                loss=self.loss,
                bucket=self.bucket,
                thread=self.thread,          # <- set to 1 for reproducibility
                verbose=self.verbose,
                label=self.label_prefix
            )
        return self

    def predict_proba(self, X):
        """
        Return class probabilities in scikit-learn's shape (n_samples, 2)
        where column 0 is P(class=0) and column 1 is P(class=1).
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call .fit(X, y) first.")

        # fastText supports batched prediction for a list of strings.
        X_list = [self._sanitize_text(t) for t in X]
        labels_list, scores_list = self.model_.predict(X_list, k=2)

        n_samples = len(X_list)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for i, (labels, scores) in enumerate(zip(labels_list, scores_list)):
            d = {int(l.replace(self.label_prefix, "")): float(s)
                 for l, s in zip(labels, scores)}
            # Falls fastText nur Top-1 zurückgibt, fehlt evtl. eine Klasse -> auffüllen
            # Binärfall: wenn nur p1 geliefert, setze p0 = 1 - p1 (und umgekehrt)
            if n_classes == 2 and (0 in d) ^ (1 in d):
                if 1 in d and 0 not in d:
                    d[0] = 1.0 - d[1]
                elif 0 in d and 1 not in d:
                    d[1] = 1.0 - d[0]

            for j, cls in enumerate(self.classes_):
                if cls in d:
                    proba[i, j] = d[cls]

        return proba

    def predict(self, X):
        """
        Thresholded prediction on P(class=1).
        """
        proba = self.predict_proba(X)
        if 1 in self.classes_:
            pos_idx = int(np.where(self.classes_ == 1)[0][0])
            return (proba[:, pos_idx] >= float(self.decision_threshold)).astype(self.classes_.dtype)
        # Fallback (multi-/other Labels): argmax
        return self.classes_[np.argmax(proba, axis=1)]
    
    def decision_function(self, X):
        # fallback: score for positive class.
        return self.predict_proba(X)[:, int(1 in self.classes_)]
    
    # in fasttext_sklearn.py, in der Klasse FastTextSklearnClassifier:

    def __getstate__(self):
        # Alles kopieren
        state = self.__dict__.copy()
        ft_bytes = None
        # fastText-Objekt (nicht pickelbar) in Bytes serialisieren
        if state.get("model_", None) is not None:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
                tmp_path = tf.name
            try:
                self.model_.save_model(tmp_path)
                with open(tmp_path, "rb") as f:
                    ft_bytes = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        # Im Pickle-Zustand: Bytes ablegen, echtes Objekt entfernen
        state["model_bytes_"] = ft_bytes
        state["model_"] = None
        return state

    def __setstate__(self, state):
        # Zustand wiederherstellen
        self.__dict__.update(state)
        ft_bytes = state.get("model_bytes_", None)
        if ft_bytes is not None:
            import tempfile, os, fasttext
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
                tmp_path = tf.name
                tf.write(ft_bytes)
                tf.flush()
            try:
                self.model_ = fasttext.load_model(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


       



