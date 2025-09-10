# eurobert_skorch.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EpochScoring, EarlyStopping, LRScheduler
from sklearn.metrics import fbeta_score, make_scorer

from src.widemlp_skorch import OptunaPruningCallbackSkorch, SeedEverything
from src.bilstm_skorch import CudaCacheClear  # reuse from BiLSTM

# EuroBERT Wrapper
class EuroBERTClassifier(nn.Module):
    def __init__(self, model_name="EuroBERT/EuroBERT-210m", num_labels=2, max_length=1000):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.max_length = max_length

    def forward(self, X, attention_mask=None):
        # Input kommt als Liste von Strings
        if isinstance(X, (list, tuple)) and isinstance(X[0], str):
            enc = self.tokenizer(
                list(X),
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            X = enc["input_ids"]
            attention_mask = enc["attention_mask"]

        outputs = self.model(input_ids=X, attention_mask=attention_mask)
        return outputs.logits

# Linear Warmup + Decay Scheduler (EuroBERT default)
def make_linear_scheduler(warmup_ratio, total_epochs):
    def lr_lambda(current_epoch):
        warmup_epochs = int(warmup_ratio * total_epochs)
        if current_epoch < warmup_epochs:
            return float(current_epoch) / max(1, warmup_epochs)
        return max(
            0.0,
            float(total_epochs - current_epoch) / max(1, total_epochs - warmup_epochs)
        )
    return lr_lambda

# Skorch Net Factory
def make_eurobert_skorch(
    model_name="EuroBERT/EuroBERT-210m",
    max_length=1000,
    num_labels=2,
    max_epochs=5,
    lr=2e-5,
    batch_size=16,
    optimizer=torch.optim.AdamW,
    criterion_fn=nn.CrossEntropyLoss,
    weight_decay=0.1,
    random_state=42,
    force_cuda=True,
    trial=None,
    use_early_stopping=True,
    es_monitor="valid_f2",
    es_patience=1,
    es_threshold=1e-4,
    es_threshold_mode="rel",
    es_lower_is_better=False
):
    device = "cuda" if (force_cuda and torch.cuda.is_available()) else "cpu"

    # Callbacks
    callbacks = []
    if trial is not None:
        callbacks.append(OptunaPruningCallbackSkorch(trial,
                                                     monitor=es_monitor,
                                                     mode="max"))
    callbacks.append(SeedEverything(seed=random_state))

    # Business metric: F2
    callbacks.append(EpochScoring(
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        lower_is_better=False,
        on_train=False,
        name="valid_f2"
    ))

    # Early stopping
    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            threshold=es_threshold,
            threshold_mode=es_threshold_mode,
            lower_is_better=es_lower_is_better
        ))

    # Scheduler: Linear with warmup (EuroBERT default)
    callbacks.append(LRScheduler(
        policy=torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=make_linear_scheduler(warmup_ratio=0.1, total_epochs=max_epochs)
    ))

    callbacks.append(CudaCacheClear())

    # Skorch Net
    net = NeuralNetClassifier(
        module=EuroBERTClassifier,
        module__model_name=model_name,
        module__num_labels=num_labels,
        module__max_length=max_length,

        criterion=criterion_fn,
        optimizer=optimizer,
        optimizer__betas=(0.9, 0.95),   # EuroBERT defaults
        optimizer__eps=1e-5,
        optimizer__weight_decay=weight_decay,  # 0.1 by default

        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        train_split=ValidSplit(0.2, stratified=True, random_state=random_state),
        device=device,
        callbacks=callbacks,
        verbose=0
    )
    return net
