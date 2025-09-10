# bilstm_skorch.py
# setup as in Adhikari et al. 2019 Rethinking Complex Neural Network Architectures for Document Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EpochScoring, EarlyStopping, Callback
from skorch.dataset import ValidSplit
from sklearn.metrics import fbeta_score, make_scorer
from src.widemlp_skorch import OptunaPruningCallbackSkorch, SeedEverything  # reuse from MLP setup

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")



# Ensures reproducibility (important for RNNs) -> 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#helper function because of torch bug
def bilstm_collate(batch):
    """
    Unterstützt zwei Batch-Formate:
      1) Mit Labels:  [ ((seq_i, mask_i), y_i), ... ]  ->  ((B,L,D),(B,L)), (B,)
      2) Ohne Labels: [ (seq_i, mask_i), ... ]          ->  ((B,L,D),(B,L))
    """
    first = batch[0]

    # Fall A: ((seq, mask), y)
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (list, tuple)):
        xs, ys = zip(*batch)              # xs: list of (seq_i, mask_i)
        seqs, masks = zip(*xs)
        seqs  = torch.as_tensor(np.stack(seqs),  dtype=torch.float32)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
        ys_np = np.asarray(ys)
        if ys_np.ndim > 1:
            ys_np = ys_np.reshape(-1)
        ys_t  = torch.from_numpy(ys_np.astype(np.int64, copy=False))
        return (seqs, masks), ys_t

    # Fall B: (seq, mask) – z.B. bei predict()/predict_proba()
    if isinstance(first, (list, tuple)) and len(first) == 2:
        seqs, masks = zip(*batch)
        seqs  = torch.as_tensor(np.stack(seqs),  dtype=torch.float32)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
        return (seqs, masks)

    raise TypeError("bilstm_collate: unerwartete Batch-Struktur")

class CudaCacheClear(Callback):
    def on_train_end(self, net, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# WeightDrop: Dropout on recurrent connections
# "We also regularize the input–hidden and hidden–hidden BiLSTM connections using embedding dropout and weight dropping, respectively"
class WeightDrop(torch.nn.Module):
    """
    Applies dropout to the hidden-to-hidden weights of RNNs (e.g. weight_hh_l0) - as in Ahdikari et al. 2019
    """
    def __init__(self, module, weights, dropout=0.0):
        super().__init__()
        self.module = module
        self.weights = tuple(weights)
        self.dropout = float(dropout)
        self._setup()

    def _setup(self):
            # Für jede versteckte Gewichtsmatrix:
            for name_w in self.weights:
                # Original-Parameter holen
                w_param = getattr(self.module, name_w)            # nn.Parameter
                if not isinstance(w_param, torch.nn.Parameter):
                    raise TypeError(f"{name_w} muss nn.Parameter sein, ist aber {type(w_param)}")

                # (a) LERNBARE Rohkopie unter <name>_raw registrieren
                #     -> diese Parameter optimiert der Optimizer
                self.module.register_parameter(name_w + "_raw", torch.nn.Parameter(w_param.data.clone()))

                # (b) Den Originalnamen wieder als Parameter registrieren,
                #     aber eingefroren (requires_grad=False). Darauf schreiben wir später .data.
                #     Vorher den alten Eintrag entfernen, damit register_parameter nicht kollidiert.
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w, torch.nn.Parameter(w_param.data.clone(), requires_grad=False))


    @torch.no_grad()
    def _setweights(self):
        # Bei jedem Forward die gedroppte Version in den eingefrorenen Parameter schreiben
        for name_w in self.weights:
            raw_w  = getattr(self.module, name_w + "_raw")         # lernbarer Roh-Parameter
            dropped = F.dropout(raw_w, p=self.dropout, training=self.training)
            tgt_param = getattr(self.module, name_w)                # eingefrorener Parameter
            # NUR die Daten kopieren (Typ bleibt nn.Parameter)
            tgt_param.data.copy_(dropped.data)

        # Optional: cuDNN Flatten aktualisieren (sicher, kostet wenig)
        if hasattr(self.module, "flatten_parameters"):
            self.module.flatten_parameters()

        # flat_weights nachziehen, damit keine None-Einträge übrig bleiben
        if hasattr(self.module, "_flat_weights"):
            self.module._flat_weights = [
                getattr(self.module, wn)
                if hasattr(self.module, wn) else p
                for wn, p in zip(self.module._flat_weights_names,
                                self.module._flat_weights)
            ]

    def train(self, mode: bool = True):
        # Trainings-/Eval-Modus synchron halten
        self.module.train(mode)
        return super().train(mode)

    def forward(self, *args, **kwargs):
        self._setweights()
        # LSTM braucht nach Gewichtsänderung ein Re-Flatten für cuDNN-Speed
        if hasattr(self.module, "flatten_parameters"):
            self.module.flatten_parameters()
        return self.module(*args, **kwargs)

    # Unbekannte Attribute an das Wrapped-LSTM weiterreichen
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Custom BiLSTM for fastText sequence input
class BiLSTMClassifier(nn.Module):
    def __init__(self, 
                 embedding_dim=300, 
                 hidden_dim=512, 
                 dropout_embedding=0.1,
                 dropout_recurrent=0.2,
                 dropout_output=0.5, 
                 out_dim=2,
                 num_layers=1, 
                 bidirectional=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # LSTM with hidden_dim=512, bidirectional, 1 layer
        # "We choose 512 hidden units for the BiLSTM models"
        # "we feed the word embeddings w1:n of a document to a single-layer BiLSTM"
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # "We regularize input–hidden connections using embedding dropout [...] with dropout rates of 0.1"
        self.input_dropout = nn.Dropout(p=dropout_embedding)  # embedding dropout

        # Dropout on pooled document vector
        # "max-pooled output is regularized using a dropout rate of 0.5"
        self.output_dropout = nn.Dropout(p=dropout_output)

        # WeightDrop for recurrent dropout
        # "regularize [...] hidden–hidden BiLSTM connections using weight dropping, [...] dropout rate of 0.2"
        # ensure for both directions
        names = ['weight_hh_l0']
        if bidirectional:
            names.append('weight_hh_l0_reverse')
        self.lstm = WeightDrop(self.lstm, weights=names, dropout=dropout_recurrent)

        # Output layer (Softmax or Sigmoid -> single-label = softmax)
        # "we feed d to a sigmoid or softmax layer over the labels" -> implicitly because we use CrossEntropyLoss
        self.out = nn.Linear(hidden_dim * self.num_directions, out_dim)

    def forward(self, X, mask=None):

        # 1) Verschachtelungen entpacken (Collate liefert oft ((X_seq, X_mask),) etc.)
        while isinstance(X, (list, tuple)) and len(X) == 1:
            X = X[0]

        # Wenn X ein (X, mask)-Paar ist, splitten
        if isinstance(X, (list, tuple)) and len(X) == 2 and mask is None:
            X, mask = X

        # Falls mask ebenfalls verschachtelt ist, entpacken
        while isinstance(mask, (list, tuple)) and len(mask) == 1:
            mask = mask[0]

        # 2) In Torch-Tensoren umwandeln (Vectorizer liefert NumPy)
        import torch
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        # 3) Auf dasselbe Device/Dtype bringen
        dev = next(self.parameters()).device
        X = X.to(dev)          # FloatTensor (B, L, D)
        mask = mask.to(dev)    # Bool/Byte/Long (B, L) -> in Bool casten
        if mask.dtype != torch.bool:
            mask = mask != 0

        # X:    torch.FloatTensor (B, L, D)
        # mask: torch.BoolTensor  (B, L)
        # LSTM forward pass with masking for padded tokens
        # "[...] extracting concatenated forward and backward word-level context vectors"
        # X: (batch_size, seq_len, embedding_dim)
        # mask: (batch_size, seq_len), 1 = valid token

        X = self.input_dropout(X)  # regularize input embeddings

        lengths = mask.sum(dim=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Initialisiere hidden state für das LSTM
        #batch_size = X.size(0)
        #device = X.device
        ##h_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
        #                batch_size, self.hidden_dim, device=device)
        #c_0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
        #                batch_size, self.hidden_dim, device=device)
        #hx = (h_0, c_0)
        packed_output, _ = self.lstm(packed)

        # Unpack to get full sequence output
        #output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Unpack with fixed total length to match the original padded input (B, L, D)
        # This guarantees output.size(1) == X.size(1) == mask.size(1)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=X.size(1)
)

        # ensure mask is on same device/dtype
        if mask.device != output.device:
            mask = mask.to(output.device)
    
        # Apply mask to remove padded positions
        output = output.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))  # so max ignores padded tokens

        # Max-over-time pooling
        h = torch.max(output, dim=1).values
        
        return self.out(self.output_dropout(h)) #feed d to output layer


def make_bilstm_skorch(
    max_epochs=30,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    batch_size=16,
    embedding_dim=300,
    hidden_dim=512,
    num_layers=1,
    dropout_embedding=0.1,
    dropout_recurrent=0.2,
    dropout_output=0.5,
    bidirectional=True,
    out_dim=2,
    criterion_fn=nn.CrossEntropyLoss,
    random_state=42,
    force_cuda=True,
    trial=None,
    # early stopping
    use_early_stopping=True,
    es_monitor="valid_f2",
    es_patience=5,
    es_threshold=1e-4,
    es_threshold_mode="rel",
    es_lower_is_better=False
):
    device = "cuda" if (force_cuda and torch.cuda.is_available()) else "cpu"

    callbacks = []
    if trial is not None:
        callbacks.append(OptunaPruningCallbackSkorch(trial,
                                                     monitor=es_monitor,
                                                     mode="max"))

    # reproducibility
    callbacks.append(SeedEverything(seed=random_state))

    # linear decay
    callbacks.append(LRScheduler(
        policy=torch.optim.lr_scheduler.LinearLR,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=max_epochs
    ))

    #f2 as business metric
    callbacks.append(EpochScoring(
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        lower_is_better=False,
        on_train=False,
        name="valid_f2"
    ))

    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            threshold=es_threshold,
            threshold_mode=es_threshold_mode,
            lower_is_better=es_lower_is_better
        ))

    # Clear CUDA cache after training
    callbacks.append(CudaCacheClear())


    net = NeuralNetClassifier(
        module=BiLSTMClassifier,
        module__embedding_dim=embedding_dim,
        module__hidden_dim=hidden_dim,
        module__num_layers=num_layers,
        module__dropout_embedding=dropout_embedding,
        module__dropout_recurrent=dropout_recurrent,
        module__dropout_output=dropout_output,
        module__bidirectional=bidirectional,
        module__out_dim=out_dim,

        criterion=criterion_fn,
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        train_split=ValidSplit(0.2, stratified=True, random_state=random_state),
        iterator_train__shuffle=True,
        iterator_train__generator=torch.Generator(device='cpu').manual_seed(random_state),
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        device=device,
        callbacks=callbacks,
        verbose=0,

        iterator_train__collate_fn=bilstm_collate,
        iterator_valid__collate_fn=bilstm_collate,
    )
    return net
