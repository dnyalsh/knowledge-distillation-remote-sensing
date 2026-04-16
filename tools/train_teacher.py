"""
Teacher Model Training Script
-------------------------------
Trains a RegNetY-032 teacher model on a remote sensing dataset
(AID, UC-Merced, NWPU-RESISC45, or Optimal-31) for use in
Knowledge Distillation experiments.

Author: Danial Shariati
University of Genoa — MSc Engineering for Natural Risk Management
Supervisor: Prof. Gabriele Moser

Usage:
    python tools/train_teacher.py --train_dir /path/to/train --val_dir /path/to/val \
        --num_classes 31 --epochs 80 --output Teacher.pt
"""

import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.models import create_model

cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description="Train teacher model for KD")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data folder")
    parser.add_argument("--val_dir",   type=str, required=True, help="Path to validation data folder")
    parser.add_argument("--model",       type=str, default="regnety_032", help="timm model name (default: regnety_032)")
    parser.add_argument("--num_classes", type=int, default=31,   help="Number of output classes")
    parser.add_argument("--img_size",    type=int, default=384,  help="Input image size")
    parser.add_argument("--batch_size",  type=int, default=20,   help="Batch size")
    parser.add_argument("--epochs",      type=int, default=80,   help="Number of training epochs")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output",      type=str, default="Teacher.pt", help="Path to save best model")
    return parser.parse_args()


def build_loaders(train_dir, val_dir, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_dir, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, predictions = model(x).max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)
    model.train()
    accuracy = 100.0 * num_correct / num_samples
    print(f"  Accuracy: {num_correct}/{num_samples} = {accuracy:.2f}%")
    return accuracy


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build data loaders
    train_loader, val_loader = build_loaders(
        args.train_dir, args.val_dir, args.img_size, args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images")

    # Load pretrained teacher model
    model = create_model(args.model, pretrained=True, num_classes=args.num_classes, global_pool="avg")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_model = copy.deepcopy(model)
    best_acc = 0.0

    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}", flush=True)

        acc = check_accuracy(val_loader, model, device)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            print(f"  >>> New best accuracy: {best_acc:.2f}%", flush=True)

    # Save best model
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(best_model.state_dict(), args.output)
    print(f"\nBest accuracy: {best_acc:.2f}%")
    print(f"Teacher model saved to: {args.output}")


if __name__ == "__main__":
    train(get_args())
