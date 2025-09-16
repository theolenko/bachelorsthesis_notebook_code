# eurobert_skorch.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EpochScoring, EarlyStopping, LRScheduler
from sklearn.metrics import fbeta_score, make_scorer
from torch.cuda.amp import autocast, GradScaler
from functools import partial

from src.widemlp_skorch import OptunaPruningCallbackSkorch, SeedEverything
from src.bilstm_skorch import CudaCacheClear  # reuse from BiLSTM

from torch.utils.data import Dataset

TOKENIZER = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-210m", trust_remote_code=True)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = list(X)
        self.y = list(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EuroBERTNet(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        # Mixed Precision Settings before super().__init__
        self.use_amp = kwargs.pop('use_amp', True)
        self.gradient_clip_val = kwargs.pop('gradient_clip_val', 1.0)
        
        # alway set scaler
        self._scaler = GradScaler() if self.use_amp else None
        
        super().__init__(*args, **kwargs)

    def infer(self, x, **fit_params):
        # for Mixed Precision: autocast for Inference
        if self.use_amp and not self.module_.training:
            with autocast():
                return self.module_(x, **fit_params)
        return self.module_(x, **fit_params)
    
    def get_iterator(self, dataset, training=False):
        # Keep standard DataLoader
        return super().get_iterator(dataset, training=training)

    def get_dataset(self, X, y=None):
        # Force use of our custom dataset always
        return TextDataset(X, y if y is not None else [0] * len(X))
    
    def train_step_single(self, batch, **fit_params):
        Xi, yi = batch
        self.module_.train()
        optimizer = self.optimizer_

        if self.use_amp and self._scaler is not None:
            # Mixed Precision Training
            with autocast():
                y_pred = self.infer(Xi, **fit_params)
                loss = self.get_loss(y_pred, yi, X=Xi, training=True)

            # Scaled Backward Pass
            self._scaler.scale(loss).backward()
            
            # Gradient Clipping with Scaler
            if self.gradient_clip_val > 0:
                self._scaler.unscale_(optimizer)  # Unscale vor Clipping!
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.gradient_clip_val)
            
            self._scaler.step(optimizer)
            self._scaler.update()

        else:
            # Standard Training (fallback)
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            
            loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.gradient_clip_val)
            
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return {"loss": loss, "y_pred": y_pred}



# EuroBERT Wrapper
class EuroBERTClassifier(nn.Module):
    def __init__(self, model_name="EuroBERT/EuroBERT-210m", num_labels=2, max_length=1000):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            trust_remote_code=True,
            torch_dtype=torch.float32 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.max_length = max_length

    def forward(self, X, attention_mask=None):
        device = next(self.model.parameters()).device

        # 1) Case: already tokenized by collate function -> (input_ids, attention_mask)
        if isinstance(X, tuple):
            input_ids, attention_mask = X
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

        # 2) Case: list of raw strings (e.g., during predict)
        elif isinstance(X, (list, tuple)) and isinstance(X[0], str):
            enc = self.tokenizer(
                list(X),
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        # 3) Case: already a tensor or list of tensors
        elif torch.is_tensor(X):
            input_ids = X.to(device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

        elif isinstance(X, list) and all(torch.is_tensor(x) for x in X):
            # list of tensors -> batch them
            input_ids = torch.stack(X).to(device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

        # 4) Unexpected input type
        else:
            raise TypeError(f"Unexpected input type for X: {type(X)}")

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Linear Warmup + Decay Scheduler (EuroBERT default)
def linear_warmup_decay(current_epoch, warmup_epochs, total_epochs):
    """Global function (picklable) for warmup + linear decay."""
    if current_epoch < warmup_epochs:
        return float(current_epoch) / max(1, warmup_epochs)
    return max(
        0.0,
        float(total_epochs - current_epoch) / max(1, total_epochs - warmup_epochs)
    )

def make_linear_scheduler(warmup_ratio, total_epochs):
    warmup_epochs = int(warmup_ratio * total_epochs)
    return partial(linear_warmup_decay,
                   warmup_epochs=warmup_epochs,
                   total_epochs=total_epochs)


def passthrough_collate(batch, max_length = 1000):
    X, y = zip(*batch)
    
    # Tokenize all texts → same length through padding
    encoded = TOKENIZER(
        list(X),
        max_length=max_length,  # use your max_length
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Now all have same length → no skorch error
    X_tokenized = encoded["input_ids"] 
    attention_mask = encoded["attention_mask"]
    y = torch.as_tensor(list(y), dtype=torch.long)
    
    return (X_tokenized, attention_mask), y

# Skorch Net Factory
def make_eurobert_skorch(
    model_name="EuroBERT/EuroBERT-210m",
    max_length=1000,
    num_labels=2,
    max_epochs=5,
    lr=2e-5,
    batch_size=2,
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
    es_lower_is_better=False,
    #mixed precision params
    use_amp=True,
    gradient_clip_val=1.0
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
    net = EuroBERTNet(
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
        verbose=0,

        #pass strings directly to forward, skip tensor conversion
        iterator_train__collate_fn=passthrough_collate,  
        iterator_valid__collate_fn=passthrough_collate,
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        iterator_train__pin_memory=False,
        iterator_valid__pin_memory=False,

        dataset=TextDataset,

        #mixed precision params
        use_amp=use_amp,
        gradient_clip_val=gradient_clip_val

        
    )
    return net
