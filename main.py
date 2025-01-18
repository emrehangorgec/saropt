import argparse
import random
import os
import time
import torch
import numpy as np
from dataset import EarthquakeDataset, MyAug
from model import MODEL_MM, MODEL_SAR, MODEL_OPT
from metrics import compute_imagewise_retrieval_metrics, compute_imagewise_f1_metrics
from logger import setup_logging

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--root", default="./data", type=str)
    parser.add_argument("--val_split", default="fold-1.txt", type=str)
    parser.add_argument("--checkpoints", default="./checkpoints", type=str)
    parser.add_argument("--sar_pretrain", default=None, type=str)
    parser.add_argument("--opt_pretrain", default=None, type=str)
    parser.add_argument(
        "--mode", default="all", type=str, choices=["all", "sar", "opt"]
    )
    args = parser.parse_args()

    logger = setup_logging(args.mode)
    logger.info(
        f"Training started with mode: {args.mode}, root: {args.root}, val_split: {args.val_split}"
    )

    train_splits = (
        ["fold-2.txt", "fold-3.txt", "fold-4.txt", "fold-5.txt"]
        if args.val_split == "fold-1.txt"
        else []
    )
    val_splits = [args.val_split]

    train_dataset = EarthquakeDataset(args.root, train_splits)
    val_dataset = EarthquakeDataset(args.root, val_splits)
    logger.info(
        f"Dataset size - Train: {len(train_dataset)}, Validation: {len(val_dataset)}"
    )

    # class weighted data sampler
    y_train = train_dataset.labels
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight, len(samples_weight)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    if args.mode == "sar":
        model = MODEL_SAR(args.sar_pretrain)
    elif args.mode == "opt":
        model = MODEL_OPT(args.opt_pretrain)
    elif args.mode == "all":
        model = MODEL_MM(args.sar_pretrain, args.opt_pretrain)
    model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    common_aug = MyAug().cuda()

    os.makedirs(args.checkpoints, exist_ok=True)
    start_time = time.time()
    best_f1 = 0
    best_auroc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        count = 0
        for i, data in enumerate(train_loader, 0):
            images = data[0]
            labels = data[1].cuda()
            sar, sarftp, opt, optftp = common_aug(
                images["sar"].cuda(),
                images["sarftp"].cuda(),
                images["opt"].cuda(),
                images["optftp"].cuda(),
            )
            optimizer.zero_grad()
            if args.mode == "sar":
                outputs = model(sar, sarftp)
            elif args.mode == "opt":
                outputs = model(opt, optftp)
            elif args.mode == "all":
                outputs = model(sar, sarftp, opt, optftp)

            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            count += 1
            train_loss += loss.item()
            if i % 10 == 0:
                logger.info(
                    f"{args.val_split}, Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}"
                )
        logger.info(f"Epoch {epoch}, Loss: {train_loss / len(train_loader)}")

        model.eval()
        out_all = []
        gt_all = []
        with torch.no_grad():
            for data in val_loader:
                images = data[0]
                labels = data[1].cuda()
                outputs = model(
                    images["opt"].cuda(),
                    images["optftp"].cuda(),
                )
                out_all.append(outputs.cpu())
                gt_all.append(labels.cpu())
        out_all = torch.cat(out_all, 0).squeeze(1)
        gt_all = torch.cat(gt_all, 0)

        f1_metrics = compute_imagewise_f1_metrics(out_all.numpy(), gt_all.numpy())
        auroc = compute_imagewise_retrieval_metrics(out_all.numpy(), gt_all.numpy())

        logger.info(
            f"Epoch {epoch}, Validation AUROC: {auroc['auroc']:.4f}, F1: {f1_metrics['f1']:.4f}"
        )

    logger.info(f"Training completed. Best F1: {best_f1}, Best AUROC: {best_auroc}")
