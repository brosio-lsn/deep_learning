from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 200

@dataclass
class TrainConfig:
    batch_size: int 
    num_epochs: int
    lr: float
    log_interval: float   # fraction of epoch
    enable_logging: bool 
    save_model: bool 
    seed: int
    out_dir: str
    exp_name: str 
