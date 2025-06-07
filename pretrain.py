import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

from model.model_config import AttentionConfig

from dataset import PretrainDataset
from config import Config
from model.model import DouDiZhu
from tokenizer.tokenizer import CustomTokenizer
from tqdm import tqdm
import deepspeed

def train_epoch(epoch, model, train_loader, optimizer, lr_scheduler):
    model.train()
    print("epoch:{}".format(epoch))
    with tqdm(train_loader) as t:
        for step, (x, labels) in enumerate(t):
            x = x.to(model.device)
            labels = labels.to(model.device)
            logits = model(x)
            mask = (labels != pad_id)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            mask = mask.view(-1)
            optimizer.zero_grad()
            loss = loss_fn(logits_flat[mask], labels_flat[mask])
            model.backward(loss)
            model.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            t.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.10f}", epoch=epoch + 1)
            if (step + 1) % train_cfg.save_interval == 0 and model.local_rank == 0:
                save_checkpoint(epoch, model, optimizer, lr_scheduler)


def save_checkpoint(epoch, model, optimizer, lr_scheduler):
    os.makedirs("./checkpoints", exist_ok=True)
    model.save_checkpoint(
        save_dir="./checkpoints",
        tag=f"epoch{epoch}",
        client_state={
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'config': train_cfg.__dict__
        }
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    deepspeed.init_distributed()
    lm_config = AttentionConfig()
    train_cfg = Config()
    torch.manual_seed(2056)
    model = DouDiZhu()
    print(f'模型总参数量{count_parameters(model)}')
    df = pd.read_csv("./data/pretrain_data.csv")
    df = df.sample(frac=1.0)
    dataset = PretrainDataset(df, max_length=lm_config.max_seq_len)
    with open("deepspeed_config.json", "r") as f:
        ds_config = json.load(f)
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=dataset
    )

    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=train_cfg.num_workers
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    pad_id = CustomTokenizer().tokenize("[PAD]")[0]
    for epoch in range(train_cfg.epochs):
        sampler.set_epoch(epoch)
        train_epoch(epoch, model_engine, train_loader, optimizer, lr_scheduler)
