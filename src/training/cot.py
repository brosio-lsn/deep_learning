import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  
from dataclasses import asdict

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems

from src.training.configs import ModelConfig, TrainConfig
from src.training.trainers import COTTrainer
from src.data.cot_dataset import CoTAdditionDataset, COT_VOCAB_TOKENS, ID2TOK
from src.data.sampler import BucketBatchSampler
from src.models.transformers import COTTransformer



def masked_cross_entropy(logits, target_ids, mask):
    """
    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool, True where we train

    Returns CE averaged over masked positions only.
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)   # (B*L, V)
    targets_flat = target_ids.reshape(-1)          # (B*L,)
    mask_flat = mask.reshape(-1)                   # (B*L,)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cfg = BoardConfig(H=4, W=5, n_digits=3)

    n_train = 40_000
    n_val = 10_000
    batch_size = 64
    num_epochs = 5
    lr = 3e-4


    train_problems = generate_diversified_problems(cfg, n_train, seed=0)
    val_problems   = generate_diversified_problems(cfg, n_val, seed=1)

    train_ds = CoTAdditionDataset(train_problems)
    val_ds   = CoTAdditionDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True))
    val_loader = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds, batch_size=batch_size, shuffle=True))

    num_training_batches = len(train_loader)
    num_val_batches = len(val_loader)

    model_cfgs = [
        ModelConfig(
            d_model=64,
            nhead=1,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=64,
            nhead=2,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=128,
            nhead=2,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        ),

        ModelConfig(
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        ),
        ModelConfig(
            d_model=256,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_len=200,
        )
    ]

    train_cfg = TrainConfig(
        batch_size = 64,
        num_epochs = 10,
        lr = 3e-4,
        log_interval = 0.1, 
        enable_docs=True,
        save_model = True,
        seed = 42,
        exp_name = "cot_addition",
        out_dir = "models"
    )

    trainer = COTTrainer(
        model=None,
        optimizer=None,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        id2tok=ID2TOK,
        train_cfg=train_cfg,
        masked_cross_entropy_fn=masked_cross_entropy
    )

    for cfg in model_cfgs:

        model_cfg = cfg

        model = COTTransformer(
            vocab_size= len(COT_VOCAB_TOKENS),
            **asdict(model_cfg)
        ).to(device)

        trainer.model = model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")
        print(f"≈ {trainable_params/1e6:.3f}M trainable parameters\n")

        print(f'Model config {model_cfg}')
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
        trainer.optimizer = optimizer
        trainer.fit()
