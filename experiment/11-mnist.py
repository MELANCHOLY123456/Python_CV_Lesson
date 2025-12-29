import os
import struct
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# IDX 读取工具
# ---------------------------
def read_idx_images(path: str) -> np.ndarray:
    """
    读取 IDX3 图像文件，返回形状 [N, 1, 28, 28]，dtype=float32，范围[0,1]
    """
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic} (expected 2051)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num, rows, cols).astype(np.float32) / 255.0
    images = images[:, None, :, :]  # add channel dim
    return images


def read_idx_labels(path: str) -> np.ndarray:
    """
    读取 IDX1 标签文件，返回形状 [N]，dtype=int64
    """
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic} (expected 2049)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    labels = data.astype(np.int64)
    if labels.shape[0] != num:
        raise ValueError("Label count mismatch.")
    return labels


# ---------------------------
# Dataset
# ---------------------------
class MNISTIdxDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str):
        self.images = read_idx_images(images_path)
        self.labels = read_idx_labels(labels_path)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("Images/labels length mismatch.")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.images[idx])          # [1,28,28], float32
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# LeNet-5 (MNIST 适配版)
# 输入: 1x28x28
# ---------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)    # 28->24  1*28*28->6*24*24
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                  # 24->12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)   # 12->8
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                  # 8->4

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------
# 训练 / 评估
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def train(
    data_dir: str = "./mnist/",
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    num_workers: int = 2,
    save_path: str = "./lenet_mnist.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images  = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels  = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    if not (os.path.exists(train_images) and os.path.exists(train_labels) and
            os.path.exists(test_images) and os.path.exists(test_labels)):
        raise FileNotFoundError(
            f"IDX files not found in {data_dir}  Expected:\n"
            f"- {train_images}\n- {train_labels}\n- {test_images}\n- {test_labels}"
        )

    # 划分 train/val（简单做法：从训练集里切一部分做验证）
    full_train = MNISTIdxDataset(train_images, train_labels)
    test_set = MNISTIdxDataset(test_images, test_labels)

    n = len(full_train)
    val_size = 5000
    train_size = n - val_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": val_acc}, save_path)
            print(f"  Saved best model to {save_path} (val_acc={val_acc:.4f})")

    # 测试
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.4f}")

    return model


# ---------------------------
# 推理示例（给一张 28x28 灰度图 numpy）
# ---------------------------
@torch.no_grad()
def predict_single(model: nn.Module, image_28x28: np.ndarray) -> int:
    """
    image_28x28: numpy, shape [28,28], dtype uint8 or float
    """
    if image_28x28.dtype != np.float32:
        image = image_28x28.astype(np.float32)
    else:
        image = image_28x28

    if image.max() > 1.0:
        image /= 255.0

    x = torch.from_numpy(image[None, None, :, :])  # [1,1,28,28]
    device = next(model.parameters()).device
    logits = model(x.to(device))
    return int(logits.argmax(dim=1).item())


if __name__ == "__main__":
    train(
        data_dir="./mnist/",
        batch_size=128,
        epochs=10,
        lr=1e-3,
        num_workers=2,
        save_path="./lenet_mnist.pth"
    )
