import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler
from model.model import DouDiZhu
from model.model_config import AttentionConfig

from config import Config
from dataset import SFTDataset
import deepspeed

from tqdm import tqdm


def train_epoch(epoch, model, train_loader, optimizer, lr_scheduler):
    model.train()
    print("epoch:{}".format(epoch))
    with tqdm(train_loader) as t:
        for step, (x, labels, mask) in enumerate(t):
            x = x.to(model.device)
            labels = labels.to(model.device)
            logits = model(x)
            mask = (mask == 1)
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
            if (step + 1) % train_cfg.save_interval == 0:
                model.save_checkpoint(
                    save_dir="./checkpoints",
                    tag=f"full_sft_epoch{epoch}",
                    client_state={
                        "epoch": epoch,  # 保存当前 epoch
                        "step": step,
                        "lr_scheduler": lr_scheduler.state_dict(),  # 保存学习率调度器状态
                    }
                )


if __name__ == "__main__":

    deepspeed.init_distributed()
    lm_config = AttentionConfig()
    train_cfg = Config()
    torch.manual_seed(2056)
    model = DouDiZhu()

    df = pd.read_csv('./data/sft_data_single.csv')
    df = df.sample(frac=1.0)
    dataset = SFTDataset(df)

    with open("deepspeed_config.json", "r") as f:
        ds_config = json.load(f)
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=dataset
    )
    model_engine.load_checkpoint(
        "./checkpoints",
        tag="epoch5",
        load_optimizer_states=False,
        load_lr_scheduler_states=False
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
    for epoch in range(0, train_cfg.epochs):
        sampler.set_epoch(epoch)
        train_epoch(epoch, model_engine, train_loader, optimizer, lr_scheduler)
