# widemlp_skorch.py
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
import optuna
from skorch.callbacks import Callback, LRScheduler, EpochScoring, EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score, make_scorer
import numpy as np
import random

#try to make everything reproducible
class SeedEverything(Callback):
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def on_train_begin(self, net, **kwargs):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

# WideMLP architecture based on Galke et al.
# Single hidden layer with 1024 ReLU units, dropout p=0.5
# Binary classification with 2 logits for CrossEntropyLoss
class WideMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=1024, dropout_p=0.5, out_dim=2,
                 activation=nn.ReLU, num_layers=1):
        super().__init__()
        # Erste Linearebene (lazy falls in_features==0 gesetzt wird)
        first = (nn.Linear(in_features, hidden_dim)
                 if in_features and in_features > 0
                 else nn.LazyLinear(hidden_dim))
        layers = [first, activation(inplace=True), nn.Dropout(p=dropout_p)]
        # Optionale zusätzliche Hidden-Layer
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       activation(inplace=True),
                       nn.Dropout(p=dropout_p)]
        # Output
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.ff = nn.Sequential(*layers)

    def forward(self, X):
        # X: dense float tensor of shape [batch_size, in_features]
        return self.ff(X)

# Factory for a sklearn-compatible skorch classifier.

def make_widemlp_skorch(
    max_epochs=100,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    batch_size=16,
    in_features=0,                      
    hidden_dim=1024,                    
    num_layers=1,                       
    dropout_p=0.5,                      
    activation_fn=nn.ReLU,              
    out_dim=2,                          #(binary default)
    criterion_fn=nn.CrossEntropyLoss,   
    random_state=42,
    force_cuda=True,
    trial=None,
    # params for early stopping
    use_early_stopping=True,
    es_monitor="valid_f2",
    es_patience=5, #how many epochs to wait before stopping
    es_threshold=1e-4, #has to improve f.e 0.01 to not stop
    es_threshold_mode="rel", #independant of scales
    es_lower_is_better=False #maximize our f2 business metric
):
    """Create WideMLP wrapped by skorch.NeuralNetClassifier.

    Args:
        max_epochs: Number of training epochs (default: 100)
        lr: Initial learning rate for Adam (default: 1e-3)
        batch_size: Mini-batch size (default: 16)
        in_features: Input dimensionality (set externally in CV, default: 0)
        hidden_dim: Width of hidden layers (default: 1024)
        num_layers: Number of hidden layers (default: 1 = WideMLP)
        dropout_p: Dropout rate (default: 0.5)
        activation_fn: Activation function (default: ReLU)
        out_dim: Output dimension (default: 2 for binary classification)
        criterion_fn: Loss function (default: CrossEntropyLoss)
        random_state: Seed for reproducibility
        force_cuda: Use GPU if available
    """
    device = "cuda" if (force_cuda and torch.cuda.is_available()) else "cpu"

    callbacks = []
    if trial is not None:
        callbacks.append(OptunaPruningCallbackSkorch(trial, 
                                                     monitor="valid_f2", 
                                                     mode="max"))

    # try to make everything reproducible
    callbacks.append(SeedEverything(random_state))

    # linear decay: linearly decaying learning rate schedule over train time
    callbacks.append(
        LRScheduler(
            policy=torch.optim.lr_scheduler.LinearLR,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max_epochs,
        )
    )
    callbacks.append(
    EpochScoring(
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        lower_is_better=False,
        on_train=False,
        name="valid_f2",
        )
    )
    #employ early stopping due to overfitting on small dataset
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=es_monitor,
                patience=es_patience,
                threshold=es_threshold,
                threshold_mode=es_threshold_mode,
                lower_is_better=es_lower_is_better,
            )
        )

    net = NeuralNetClassifier(
        module=WideMLP,
        module__in_features=in_features,
        module__hidden_dim=hidden_dim,
        module__num_layers=num_layers,
        module__dropout_p=dropout_p,
        module__activation=activation_fn,
        module__out_dim=out_dim,

        criterion=criterion_fn,
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        iterator_train__shuffle=True,
        iterator_train__generator=torch.Generator(device='cpu').manual_seed(random_state),
        train_split=ValidSplit(0.2, stratified=True, random_state=random_state),
        device=device,
        callbacks=callbacks,
        verbose=0,
    )
    return net

# TODO Maybe need adoption here: as it stands - mlp_obkjective AND biLSTM_objective both access this class
class OptunaPruningCallbackSkorch(Callback):
    """
    Reports a monitored metric to Optuna at the end of each epoch and
    prunes unpromising trials. Use with skorch + train_split=ValidSplit(...).
    monitor: a key from net.history (e.g., 'valid_loss', 'valid_acc', or custom)
    mode: 'min' for losses, 'max' for scores
    """
    def __init__(self, trial: optuna.trial.Trial, monitor: str = "valid_f2", mode: str = "max", step_base: int = 0, logger=None):
        assert mode in {"min", "max"}
        self.trial, self.monitor, self.mode = trial, monitor, mode
        self.step_base = step_base
        self._epoch_idx = -1
        self.logger = logger

    def on_train_begin(self, net, **kwargs):
        self._epoch_idx = -1  # reset per fit()

    def on_epoch_end(self, net, **kwargs):
        row = net.history[-1]
        if self.monitor not in row:
            return
        self._epoch_idx += 1
        step = self.step_base + self._epoch_idx

        #  best-so-far
        hist_vals = [float(h[self.monitor]) for h in net.history if self.monitor in h]
        if not hist_vals:
            return
        if self.mode == "max":
            value = max(hist_vals)
        else:
            value = min(hist_vals)

        self.trial.report(value, step=step)
        if self.trial.should_prune():
            if self.logger:
                self.logger.info(
                    f"Trial {self.trial.number}: PRUNED at epoch {self._epoch_idx} "
                    f"({self.monitor}={value:.6f}, step={step})"
                )
            raise optuna.TrialPruned(
                f"Pruned at epoch {self._epoch_idx}: {self.monitor}={value:.6f}"
            )

# introduce into pipeline to ensure float32 input (dense vectors) --> necessary for mlp
class ToFloat32Dense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        # mark as fitted so Pipeline/check_is_fitted is satisfied
        try:
            self.n_features_in_ = int(X.shape[1])  # optional but nice to have
        except Exception:
            self.n_features_in_ = None
        self.fitted_ = True
        return self
    
    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)