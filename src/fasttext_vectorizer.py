import os, re, numpy as np, torch
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy



_KV_CACHE = {}  # key: (path, mode) -> KeyedVectors  (or FastTextKeyedVectors)

try:
    from gensim.models.fasttext import load_facebook_vectors  # .bin with Subword/OOV
    HAS_FB_BIN = True
except Exception:
    HAS_FB_BIN = False
from gensim.models import KeyedVectors  # .vec fallback (no Subword)

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for Facebook fastText .bin/.vec.
    - mode="pool": outputs a single doc embedding (n, 300).
    - mode="sequence": outputs padded sequences for BiLSTM (X_seq, X_mask).
    """
    def __init__(self,
                 lang: str = "de",
                 model_path: Optional[str] = None,
                 model_dir: str = "src/fasttext_embeddings",
                 mode: str = "pool",                 # "pool" | "sequence"
                 pool: str = "mean",                 # "mean" | "max" | "tfidf" only relevant when mode: pool
                 max_len: int = 1000,                 # only used for mode="sequence", see our doc segmentation strategy
                 lowercase: bool = True,
                 token_pattern: str = r"(?u)\b\w\w+\b",
                 tfidf_max_features: Optional[int] = 200000,
                 debug:bool = False,
                 cache_key: Optional[str] = None):
        
        self.cache_key = cache_key
        self.debug = debug
        self.lang = lang
        self.model_path = model_path
        self.model_dir = model_dir
        self.mode = mode
        self.pool = pool
        self.max_len = max_len
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.tfidf_max_features = tfidf_max_features

        self._kv = None
        self._dim = None
        self._tfidf = None
        self._tokenizer = re.compile(token_pattern)

    def _resolve_path(self) -> str:
        if self.model_path:
            return self.model_path
        fname = f"cc.{self.lang}.300.bin"
        return os.path.join(self.model_dir, fname)

    #same tokenization as with tfidf default
    def _simple_tok(self, text: str) -> List[str]:
        if self.lowercase: 
            text = text.lower()
        return self._tokenizer.findall(text)
    
    
    def fit(self, X: List[str], y=None):
        path = self._resolve_path()

        # 0) re-use if already injected or previously fit
        if self._kv is not None:
            if self.debug: print("[FastTextVectorizer] reuse already-set _kv")
            return self

        # 1) process-wide cache key (stable across clones)
        key = (self.cache_key or path, self.mode)

        # 2) hit cache?
        if key in _KV_CACHE:
            self._kv = _KV_CACHE[key]
            self._dim = int(self._kv.vector_size)
            if self.debug: print(f"[FastTextVectorizer] CACHE HIT for {key}")
        else:
            # 3) load once (original behavior) and put into cache
            if HAS_FB_BIN and path.endswith(".bin"):
                if self.debug: print(f"[FastTextVectorizer] LOADING .bin from {path}")
                self._kv = load_facebook_vectors(path)
            else:
                if self.debug: print(f"[FastTextVectorizer] LOADING .vec from {path}")
                self._kv = KeyedVectors.load_word2vec_format(path, binary=False)
            self._dim = int(self._kv.vector_size)
            _KV_CACHE[key] = self._kv
            if self.debug: print(f"[FastTextVectorizer] CACHED under {key}")

        # (TF-IDF pooling))
        if self.mode == "pool" and self.pool == "tfidf":
            self._tfidf = TfidfVectorizer(lowercase=self.lowercase,
                                        token_pattern=self.token_pattern,
                                        max_features=self.tfidf_max_features,
                                        dtype=np.float32).fit(X)
        return self
    

    def _word_vec(self, tok: str) -> Optional[np.ndarray]:
        # .bin: get_vector works for OOV; .vec: only known words
        try:
            return self._kv.get_vector(tok, norm=False)
        except KeyError:
            return None

    def _pool_doc(self, toks: List[str], tfmap=None) -> np.ndarray:
        vecs, weights = [], []
        for t in toks:
            v = self._word_vec(t)
            if v is None: 
                continue
            if self.pool == "tfidf" and tfmap is not None:
                w = tfmap.get(t, 0.0)
                if w > 0: 
                    vecs.append(v); weights.append(w)
            else:
                vecs.append(v)
        if not vecs:
            return np.zeros(self._dim, dtype=np.float32)
        V = np.vstack(vecs).astype(np.float32)
        if self.pool == "max":
            return V.max(axis=0)
        if self.pool == "tfidf" and weights:
            W = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
            return (V * W).sum(axis=0) / (W.sum() + 1e-9)
        return V.mean(axis=0)

    def transform(self, X: List[str]):
        if self.mode == "pool":
            # Precompute TF-IDF maps if required
            tfmaps = None
            if self.pool == "tfidf" and self._tfidf is not None:
                Xtf = self._tfidf.transform(X)
                vocab = self._tfidf.vocabulary_
                inv = {i: t for t, i in vocab.items()}
                tfmaps = []
                for i in range(Xtf.shape[0]):
                    row = Xtf.getrow(i)
                    tfmaps.append({inv[j]: float(w) for j, w in zip(row.indices, row.data)})
            out = np.zeros((len(X), self._dim), dtype=np.float32)
            for i, doc in enumerate(X):
                toks = self._simple_tok(doc)
                tfmap = tfmaps[i] if tfmaps is not None else None
                out[i] = self._pool_doc(toks, tfmap)
            return out

        # mode == "sequence": pad/truncate to fixed length
        n, L, d = len(X), self.max_len, self._dim
        X_seq = np.zeros((n, L, d), dtype=np.float32)
        X_mask = np.zeros((n, L), dtype=bool)
        for i, doc in enumerate(X):
            toks = self._simple_tok(doc)
            cur = 0
            for t in toks:
                if cur >= L: break
                v = self._word_vec(t)
                if v is None: continue
                X_seq[i, cur, :] = v
                X_mask[i, cur] = True
                cur += 1
        # Return as tuple: skorch modules can consume (X_seq, X_mask) , as torch tensors
        #return (torch.from_numpy(X_seq), torch.from_numpy(X_mask))
        return (X_seq, X_mask)

    # caching of loaded embeddings for more efficient reuse
    def set_preloaded_vectors(self, kv):
        """
        Allow injection of already loaded KeyedVectors to avoid re-loading .bin file.
        """
        self._kv = kv
        self._dim = int(kv.vector_size)
        self._fitted = True

    #for more efficient caching
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_kv":
                setattr(result, k, v)  # shared reference (OK, read-only)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

