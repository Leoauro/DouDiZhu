from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 16
    num_workers: int = 1
    epochs: int = 10
    lr: float = 3e-4
    save_interval: int = 10000
