import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split
import copy


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset

BATCH_SIZE = 512
# BATCH_SIZE = 768
baseimgdir = "./input/6/EuroSAT"

load_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # float32 in [0,1], CHW
])

def loadsplit(csvpath):
    df = pd.read_csv(csvpath)
    Xlist, ylist = [], []
    for _, row in df.iterrows():
        imgrelpath = row["Filename"]
        label = int(row["Label"])
        imgpath = os.path.join(baseimgdir, imgrelpath)
        img = Image.open(imgpath).convert("RGB")
        img = load_transform(img)
        Xlist.append(img)
        ylist.append(label)
    X = torch.stack(Xlist)
    y = torch.tensor(ylist, dtype=torch.long)
    return X, y

trainX, trainy = loadsplit("./input/6/EuroSAT/train.csv")
valX,   valy   = loadsplit("./input/6/EuroSAT/validation.csv")
testX,  testy  = loadsplit("./input/6/EuroSAT/test.csv")

mean = trainX.mean(dim=(0, 2, 3))
std  = trainX.std(dim=(0, 2, 3)).clamp_min(1e-6)

train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    # transforms.Normalize(mean=mean, std=std),
])

# 4) Eval-time: only normalization (no randomness)
eval_transform = transforms.Compose([
    # transforms.Normalize(mean=mean, std=std),
])

# 5) Dataset wrapper that applies transforms per sample (per epoch/batch)
class AugmentedTensorDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

        # Add this so existing code that expects TensorDataset keeps working
        self.tensors = (self.X, self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


train = AugmentedTensorDataset(trainX, trainy, transform=train_transform)
val   = AugmentedTensorDataset(valX,   valy,   transform=eval_transform)
test  = AugmentedTensorDataset(testX,  testy,  transform=eval_transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False)

EXPERT_CFG = {
    'path1': {
        'conv1': slice(0,   171),
        'conv2': slice(0,   171),
        'conv3': slice(0,   171),
        'conv4': slice(0,   171),
        'conv5': slice(0,   171),
        'conv6': slice(0,   171),
        'conv7': slice(0,   171),
    },
    'path2': {
        'conv1': slice(171, 342),
        'conv2': slice(171, 342),
        'conv3': slice(171, 342),
        'conv4': slice(171, 342),
        'conv5': slice(171, 342),
        'conv6': slice(171, 342),
        'conv7': slice(171, 342),
    },
    'path3': {
        'conv1': slice(342, 512),
        'conv2': slice(342, 512),
        'conv3': slice(342, 512),
        'conv4': slice(342, 512),
        'conv5': slice(342, 512),
        'conv6': slice(342, 512),
        'conv7': slice(342, 512),
    },
}
EXPERT_NAMES = list(EXPERT_CFG.keys())



class StrictGatedCNN(nn.Module):
    def __init__(self, device, num_classes=10, mean=None, std=None):
        super().__init__()
        self.device = device

        if mean is None:
            mean = torch.zeros(3)
        if std is None:
            std = torch.ones(3)

        # mean/std are already tensors in your script (computed from trainX). [file:252]
        self.register_buffer("img_mean", mean.detach().clone().float().view(1, 3, 1, 1))
        self.register_buffer("img_std",  std.detach().clone().float().view(1, 3, 1, 1))

        # ---------- 5-block CNN backbone (wider) ----------
        self.conv1 = nn.Conv2d(3,   512, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7   = nn.BatchNorm2d(512)

        self.pool_block  = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512  # still 512 after global_pool

        # ---------- gating ----------
        in_flat = 64 * 64 * 3
        self.gate_fc1_train = nn.Linear(in_flat + num_classes, 64)
        self.gate_fc2_train = nn.Linear(64, 3)
        self.gate_fc1_inf   = nn.Linear(in_flat, 64)
        self.gate_fc2_inf   = nn.Linear(64, 3)

        # ---------- classifier head (shared) ----------
        self.fc1 = nn.Linear(feat_dim, 1024)   # 512 -> 1024
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # ---------- path masks over conv filters ----------
        self._create_path_masks()

    def _create_path_masks(self):
        path_filter_ranges = EXPERT_CFG

        for path_name, cfg in path_filter_ranges.items():
            # conv1: (512, 3, 3, 3)
            conv1_mask = torch.zeros_like(self.conv1.weight, device=self.device)
            conv1_mask[cfg['conv1'], :, :, :] = 1.0
            self.register_buffer(f'conv1_mask_{path_name}', conv1_mask)

            # conv2: (512, 512, 3, 3)
            conv2_mask = torch.zeros_like(self.conv2.weight, device=self.device)
            conv2_mask[cfg['conv2'], cfg['conv1'], :, :] = 1.0
            self.register_buffer(f'conv2_mask_{path_name}', conv2_mask)

            # conv3
            conv3_mask = torch.zeros_like(self.conv3.weight, device=self.device)
            conv3_mask[cfg['conv3'], cfg['conv2'], :, :] = 1.0
            self.register_buffer(f'conv3_mask_{path_name}', conv3_mask)

            # conv4
            conv4_mask = torch.zeros_like(self.conv4.weight, device=self.device)
            conv4_mask[cfg['conv4'], cfg['conv3'], :, :] = 1.0
            self.register_buffer(f'conv4_mask_{path_name}', conv4_mask)

            # conv5
            conv5_mask = torch.zeros_like(self.conv5.weight, device=self.device)
            conv5_mask[cfg['conv5'], cfg['conv4'], :, :] = 1.0
            self.register_buffer(f'conv5_mask_{path_name}', conv5_mask)

            # conv6
            conv6_mask = torch.zeros_like(self.conv6.weight, device=self.device)
            conv6_mask[cfg['conv6'], cfg['conv5'], :, :] = 1.0
            self.register_buffer(f'conv6_mask_{path_name}', conv6_mask)

            # conv7
            conv7_mask = torch.zeros_like(self.conv7.weight, device=self.device)
            conv7_mask[cfg['conv7'], cfg['conv6'], :, :] = 1.0
            self.register_buffer(f'conv7_mask_{path_name}', conv7_mask)

        self.path_filter_ranges = path_filter_ranges


    def _forward_blocks(self, x, path_name=None):
        # get weights or masked weights (as you already do)
        if path_name is None:
            w1 = self.conv1.weight
            w2 = self.conv2.weight
            w3 = self.conv3.weight
            w4 = self.conv4.weight
            w5 = self.conv5.weight
            w6 = self.conv6.weight
            w7 = self.conv7.weight
        else:
            conv1_mask = getattr(self, f'conv1_mask_{path_name}')
            conv2_mask = getattr(self, f'conv2_mask_{path_name}')
            conv3_mask = getattr(self, f'conv3_mask_{path_name}')
            conv4_mask = getattr(self, f'conv4_mask_{path_name}')
            conv5_mask = getattr(self, f'conv5_mask_{path_name}')
            conv6_mask = getattr(self, f'conv6_mask_{path_name}')
            conv7_mask = getattr(self, f'conv7_mask_{path_name}')

            w1 = self.conv1.weight * conv1_mask
            w2 = self.conv2.weight * conv2_mask
            w3 = self.conv3.weight * conv3_mask
            w4 = self.conv4.weight * conv4_mask
            w5 = self.conv5.weight * conv5_mask
            w6 = self.conv6.weight * conv6_mask
            w7 = self.conv7.weight * conv7_mask

        # Input: 64x64
        # block1 -> 32x32
        x = F.conv2d(x, w1, self.conv1.bias, padding=1)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool_block(x)

        # block2 -> 16x16
        x = F.conv2d(x, w2, self.conv2.bias, padding=1)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool_block(x)

        # block3 -> 8x8
        x = F.conv2d(x, w3, self.conv3.bias, padding=1)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.pool_block(x)

        # block4 -> 4x4
        x = F.conv2d(x, w4, self.conv4.bias, padding=1)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.pool_block(x)

        # block5 -> 2x2
        x = F.conv2d(x, w5, self.conv5.bias, padding=1)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.pool_block(x)

        # block6: stay 2x2 (no pool)
        x = F.conv2d(x, w6, self.conv6.bias, padding=1)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)

        # block7: stay 2x2 (no pool)
        x = F.conv2d(x, w7, self.conv7.bias, padding=1)
        x = self.bn7(x)
        x = F.relu(x, inplace=True)

        return x  # (B,512,2,2) -> global_pool -> (B,512,1,1)


    def _head_from_slice(self, f):
        """
        f: (B, C, H, W) for a single path slice at conv4.
        Zero-pad to 512 and run through shared FC head.
        """
        p = self.global_pool(f)        # (B,C,1,1)
        v = torch.flatten(p, 1)        # (B,C)
        B, C = v.shape
        if C < 512:
            pad = torch.zeros(B, 512 - C, device=v.device, dtype=v.dtype)
            v512 = torch.cat([v, pad], dim=1)  # (B,512)
        else:
            v512 = v
        h = F.relu(self.fc1(v512))
        dropout_p = 0.2
        h = F.dropout(h, p=dropout_p, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=dropout_p, training=self.training)
        return self.fc3(h)             # (B,num_classes)

    def forward(self, x, labels=None, path_name=None):
        # Gate should see raw pixels; conv trunk should see normalized. [file:252]
        x_raw = x
        x = (x - self.img_mean) / self.img_std

        batchsize = x.size(0)
        flat_x = x_raw.view(batchsize, -1)   # <-- gate uses raw pixels

        # ----- gating -----
        if self.training and labels is not None:
            one_hot = F.one_hot(labels, 10).float()
            gate_input = torch.cat([flat_x, one_hot], dim=1)
            gate_fc1, gate_fc2 = self.gate_fc1_train, self.gate_fc2_train
        else:
            gate_input = flat_x
            gate_fc1, gate_fc2 = self.gate_fc1_inf, self.gate_fc2_inf

        gate_logits = F.relu(gate_fc1(gate_input))
        gate_weights = F.softmax(gate_fc2(gate_logits), dim=1)  # (B,3)

        # ----- explicit path mode (for expert FL) -----
        if path_name is not None:
            feats = self._forward_blocks(x, path_name=path_name)   # masked convs on all convs
            cfg_p = self.path_filter_ranges[path_name]
            f = feats[:, cfg_p['conv4'], :, :]                    # only this path’s conv4 channels
            logits = self._head_from_slice(f)
            return F.log_softmax(logits, dim=1), gate_weights

        # ----- normal global forward: full convs, then slice conv4 and blend -----
        feats = self._forward_blocks(x, path_name=None)           # full convs

        cfg = self.path_filter_ranges
        f1 = feats[:, cfg['path1']['conv7'], :, :]
        f2 = feats[:, cfg['path2']['conv7'], :, :]
        f3 = feats[:, cfg['path3']['conv7'], :, :]

        out1 = self._head_from_slice(f1)
        out2 = self._head_from_slice(f2)
        out3 = self._head_from_slice(f3)

        w1 = gate_weights[:, 0:1]
        w2 = gate_weights[:, 1:2]
        w3 = gate_weights[:, 2:3]

        logits = w1 * out1 + w2 * out2 + w3 * out3
        return F.log_softmax(logits, dim=1), gate_weights

# HELPER FUNCTIONS
def evaluate_model_gpu(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            pred = torch.max(logits, 1)[1]
            correct += (pred == y_batch).sum().item()
    return 100. * correct / len(test_loader.dataset)

@torch.no_grad()
def evaluate_global_per_expert(global_model, loader, device):
    """
    Evaluate final global model on the full loader, forcing each path separately.
    Returns dict: {'path1': acc%, 'path2': acc%, 'path3': acc%}.
    """
    global_model.eval()
    global_model.to(device)
    results = {}

    for path_name in ["path1", "path2", "path3"]:
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # force explicit expert path
            log_probs, _ = global_model(X_batch, y_batch, path_name=path_name)
            preds = log_probs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        acc = 100.0 * correct / total
        results[path_name] = acc
    return results

def create_data_shard_clients(base_model, device, num_clients=3):
    """
    Create num_clients clients as random data shards of the full train set.
    Each client sees all labels 0–9, just different samples.
    """
    n = len(train)
    indices = torch.randperm(n)
    shard_size = n // num_clients
    
    clients = {}
    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else n
        client_indices = indices[start:end]
        
        client_dataset = torch.utils.data.Subset(train, client_indices)
        client_loader = torch.utils.data.DataLoader(
            client_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        
        client_model = copy.deepcopy(base_model)
        clients[f'client{i+1}'] = {
            'model': client_model,
            'loader': client_loader
        }
    return clients

# def create_data_shard_clients(base_model, device, num_clients=9):
#     clients = {}
#     n = len(train)
#     indices = torch.randperm(n)
#     shard_size = n // num_clients

#     for i in range(num_clients):
#         start = i * shard_size
#         end = (i + 1) * shard_size if i < num_clients - 1 else n
#         client_indices = indices[start:end]
#         client_dataset = torch.utils.data.Subset(train, client_indices)
#         client_loader = torch.utils.data.DataLoader(
#             client_dataset, batch_size=BATCH_SIZE, shuffle=True
#         )

#         client_model = copy.deepcopy(base_model).to(device)

#         # freeze gate; only experts (conv) + head will train locally
#         set_gate_trainable(client_model, False)
#         set_conv_trainable(client_model, True)
#         set_head_trainable(client_model, True)
#         freeze_bn_affine(client_model)   # optional

#         clients[f"client{i+1}"] = {
#             "model": client_model,
#             "loader": client_loader,
#         }

#     return clients

def create_full_data_clients(base_model, device, num_clients=3):
    """
    Create num_clients clients, each seeing the *entire* train dataset.
    This approximates IID with full overlap: all clients have the same data.
    """
    clients = {}
    full_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True
    )

    for i in range(num_clients):
        client_model = copy.deepcopy(base_model)
        clients[f'client{i+1}'] = {
            'model': client_model,
            'loader': full_loader,  # all clients share same dataset
        }
    return clients


def train_one_client_epoch_label_paths(model, loader, device, optimizer, criterion):
    """
    Within a client, route samples by label group to the proper path:
    0–2 → client1 path; 3–5 → client2 path; 6–9 → client3 path.
    """
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        mask_0_2 = (y_batch >= 0) & (y_batch <= 2)
        mask_3_5 = (y_batch >= 3) & (y_batch <= 5)
        mask_6_9 = (y_batch >= 6) & (y_batch <= 9)
        
        optimizer.zero_grad()
        total_count = 0
        
        # 0–2 → Path0 (client1 mask)
        if mask_0_2.any():
            X_sub = X_batch[mask_0_2]
            y_sub = y_batch[mask_0_2]
            logits, _ = model(X_sub, y_sub, path_name='path1')
            loss_sub = criterion(logits, y_sub)
            loss_sub.backward()
            total_count += len(y_sub)
        
        # 3–5 → Path1 (client2 mask)
        if mask_3_5.any():
            X_sub = X_batch[mask_3_5]
            y_sub = y_batch[mask_3_5]
            logits, _ = model(X_sub, y_sub, path_name='path2')
            loss_sub = criterion(logits, y_sub)
            loss_sub.backward()
            total_count += len(y_sub)
        
        # 6–9 → Path2 (client3 mask)
        if mask_6_9.any():
            X_sub = X_batch[mask_6_9]
            y_sub = y_batch[mask_6_9]
            logits, _ = model(X_sub, y_sub, path_name='path3')
            loss_sub = criterion(logits, y_sub)
            loss_sub.backward()
            total_count += len(y_sub)
        
        if total_count > 0:
            optimizer.step()

def analyze_final_gating(model, test_loader, device):
    model.eval()
    gate_stats = {i: np.zeros(3) for i in range(10)}
    class_counts = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, gate_weights = model(X_batch)
            for i, label in enumerate(y_batch):
                cls = label.item()
                class_counts[cls] += 1
                gate_stats[cls] += gate_weights[i].cpu().numpy()
    
    print("\n=== FINAL GATING PATH USAGE PER CLASS ===")
    for cls in range(10):
        if class_counts[cls] > 0:
            avg_gates = gate_stats[cls] / class_counts[cls]
            path = np.argmax(avg_gates)
            print(
                f"Class {cls}: Path{path} ({avg_gates[path]:.3f}) | "
                f"P0:{avg_gates[0]:.3f} P1:{avg_gates[1]:.3f} P2:{avg_gates[2]:.3f}"
            )

def is_bn_key(key: str) -> bool:
    # covers running stats and affine params of all BN layers
    return (
        ".bn" in key and (
            key.endswith("running_mean") or
            key.endswith("running_var") or
            key.endswith("weight") or
            key.endswith("bias")
        )
    )

def is_gate_key(key: str) -> bool:
    return key.startswith("gate_fc")

@torch.no_grad()
def evaluate_client_model(client_model, loader, device):
    client_model.eval()
    client_model.to(device)
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits, _ = client_model(X_batch, y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return 100.0 * correct / total

def evaluate_client_model_for_path(client_model, loader, device, expert_name):
    client_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = client_model(X_batch, y_batch, path_name=expert_name)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_global_by_label_paths(model, loader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = torch.empty_like(y_batch)  # will be fully filled by the 3 masks

        for path_name, (lo, hi) in [
            ("path1", (0, 2)),
            ("path2", (3, 5)),
            ("path3", (6, 9)),
        ]:
            mask = (y_batch >= lo) & (y_batch <= hi)
            if not mask.any():
                continue

            X_sub = X_batch[mask]
            y_sub = y_batch[mask]

            # IMPORTANT: use the correct keyword: pathname [file:254]
            log_probs, _ = model(X_sub, y_sub, path_name=path_name)
            preds[mask] = log_probs.argmax(dim=1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return 100.0 * correct / total

def get_client_expert_for_round(client_name, round_idx):
    """
    Simple round-robin: client i trains expert (round_idx + i) % 3
    client_name is 'client1','client2','client3'.
    """
    client_idx = int(client_name.replace('client', '')) - 1  # 0,1,2
    expert_idx = (round_idx + client_idx) % len(EXPERT_NAMES)
    return EXPERT_NAMES[expert_idx]

def set_gate_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if name.startswith("gate_fc"):
            p.requires_grad = trainable

def set_conv_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if name.startswith("conv"):
            p.requires_grad = trainable

def set_head_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if name.startswith("fc"):
            p.requires_grad = trainable

def freeze_bn_affine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

def is_personalized(key):
    return "fc" in key or "gate" in key or ".bn" in key


def federated_expert_rotation(path_clients, global_model, device, num_rounds=3):
    """
    Step 3: In each round, every client trains ALL three experts (path1,path2,path3)
    sequentially, starting each expert phase from the same global state.
    The server then aggregates per-expert conv slices across clients.
    """
    criterion = nn.CrossEntropyLoss()
    FL_LR = 1e-4
    LOCAL_FL_EPOCHS = 1

    for rnd in range(num_rounds):
        print(f"\nExpert FedAvg Round {rnd+1}/{num_rounds}")
        global_state = global_model.state_dict()
        # collect updates separately for each expert
        expert_updates = {name: [] for name in EXPERT_NAMES}

        for client_name, client_info in path_clients.items():
            client_model = client_info["model"].to(device)
            loader = client_info["loader"]

            # print(f"  Round {rnd+1}, training all paths on {client_name}")

            # ====== loop over all experts on this client ======
            for path_idx, expert_name in enumerate(EXPERT_NAMES):  # ["path1","path2","path3"]
                # print(f"    {client_name}: training {expert_name}")

                # start from the same global state for each expert phase
                client_model.load_state_dict(global_state)

                # freeze everything
                for name, param in client_model.named_parameters():
                    param.requires_grad = False

                # train only higher conv layers (or all convs if you prefer)
                for layer_name in ["conv3", "conv4", "conv5", "conv6", "conv7"]:
                    layer = getattr(client_model, layer_name)
                    layer.weight.requires_grad = True

                # BN: update running stats but keep weights/bias frozen
                for m in client_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
                        if m.weight is not None:
                            m.weight.requires_grad = False
                        if m.bias is not None:
                            m.bias.requires_grad = False

                optimizer = torch.optim.Adam(
                    [p for p in client_model.parameters() if p.requires_grad],
                    lr=FL_LR,
                )

                # OPTIONAL: filter to samples where gate chooses this path
                target_idx = ["path1", "path2", "path3"].index(expert_name)
                client_model.train()

                for epoch in range(LOCAL_FL_EPOCHS):
                    for X_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                        # gate-based filtering
                        with torch.no_grad():
                            _, gate_weights = client_model(X_batch, y_batch)
                            chosen_paths = gate_weights.argmax(dim=1)  # 0,1,2

                        mask = (chosen_paths == target_idx)
                        if not mask.any():
                            continue

                        X_sub = X_batch[mask]
                        y_sub = y_batch[mask]

                        optimizer.zero_grad()
                        log_probs, _ = client_model(X_sub, y_sub, path_name=expert_name)
                        loss = criterion(log_probs, y_sub)
                        loss.backward()
                        optimizer.step()

                # collect this client's updated slices for THIS expert
                cs = client_model.state_dict()
                cfg = EXPERT_CFG[expert_name]
                expert_update = {}
                for layer_name in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7"]:
                    s = cfg[layer_name]
                    w_key = f"{layer_name}.weight"
                    expert_update[w_key] = cs[w_key][s].clone()

                expert_updates[expert_name].append(expert_update)

        # ====== server aggregation: only conv slices for each expert ======
        new_global_state = copy.deepcopy(global_state)

        for expert_name, client_states_list in expert_updates.items():
            if not client_states_list:
                continue
            cfg = EXPERT_CFG[expert_name]
            for layer_name in ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7"]:
                s = cfg[layer_name]
                w_key = f"{layer_name}.weight"
                weight_avg = sum(cs[w_key] for cs in client_states_list) / len(client_states_list)
                new_global_state[w_key][s] = weight_avg

        global_model.load_state_dict(new_global_state)

        # track global performance after this full 3-path update round
        val_acc = evaluate_model_gpu(global_model, val_loader, device)
        print(f"Expert rotation round {rnd+1} val accuracy: {val_acc:.2f}%")
        test_acc = evaluate_model_gpu(global_model, test_loader, device)
        print(f"Expert rotation round {rnd+1} test accuracy: {test_acc:.2f}%")

    return global_model

def federated_head_finetune(path_clients, global_model, device, num_rounds=5):
    criterion = nn.CrossEntropyLoss()
    FLLR = 5e-4
    LOCAL_FL_EPOCHS = 1

    for rnd in range(num_rounds):
        print(f"Head-only fine-tune round {rnd+1}/{num_rounds}")
        global_state = global_model.state_dict()
        client_states = {}

        for client_name, client_info in path_clients.items():
            client_model = client_info["model"].to(device)
            loader = client_info["loader"]

            # Load global model
            client_model.load_state_dict(global_state)

            # Freeze conv + BN + gate; train only fc1, fc2, fc3
            for name, param in client_model.named_parameters():
                if name.startswith("fc1") or name.startswith("fc2") or name.startswith("fc3"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            print_param_stats(client_model, f"Head-only FL round {rnd+1}, {client_name}")

            optimizer = torch.optim.Adam(
                [p for p in client_model.parameters() if p.requires_grad],
                lr=FLLR,
            )
            client_model.train()

            for epoch in range(LOCAL_FL_EPOCHS):
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    # normal gated forward, but gate is frozen
                    logits, gate_weights = client_model(X_batch)
                    loss_cls = criterion(logits, y_batch)
                    loss = loss_cls            # no gate loss
                    loss.backward()
                    optimizer.step()

            # collect only head params
            cs = client_model.state_dict()
            head_update = {}
            keys_to_avg = [
                "fc1.weight", "fc1.bias",
                "fc2.weight", "fc2.bias",
                "fc3.weight", "fc3.bias",
            ]
            for key in keys_to_avg:
                head_update[key] = cs[key].clone()

            num_elements = sum(v.numel() for v in head_update.values())
            bytes_sent = num_elements * 4  # float32
            print(f"  {client_name} uploads ~{bytes_sent/1024:.1f} KB for head in this round")

            client_states[client_name] = head_update

        # FedAvg on head params
        new_global_state = copy.deepcopy(global_state)
        keys_to_avg = list(next(iter(client_states.values())).keys())
        num_clients = len(path_clients)
        for key in keys_to_avg:
            if is_bn_key(key):
                continue
            new_global_state[key] = sum(client_states[c][key] for c in path_clients.keys()) / num_clients
        global_model.load_state_dict(new_global_state)


        acc = evaluate_model_gpu(global_model, test_loader, device)
        print(f"Head-only fine-tune round {rnd+1} accuracy {acc:.2f}")

    return global_model

def federated_gate_finetune(path_clients, global_model, device, num_rounds=2):
    criterion = nn.CrossEntropyLoss()
    FLLR = 5e-4
    LOCAL_FL_EPOCHS = 1
    LAMBDAGATE = 0.0      # disable class→path MSE
    ENTROPY_ALPHA = 1e-3  # same scale as in local training

    for rnd in range(num_rounds):
        print(f"\n[Gate-only] fine-tune round {rnd+1}/{num_rounds}")
        global_state = global_model.state_dict()
        client_states = {}

        for client_name, client_info in path_clients.items():
            client_model = client_info["model"].to(device)
            loader = client_info["loader"]

            client_model.load_state_dict(global_state)

            for name, param in client_model.named_parameters():
                if name.startswith("gate_fc"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.Adam(
                [p for p in client_model.parameters() if p.requires_grad],
                lr=FLLR,
            )

            client_model.train()
            for epoch in range(LOCAL_FL_EPOCHS):
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()

                    logits, gate_weights = client_model(X_batch)
                    loss_cls = criterion(logits, y_batch)
                    gate_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=1).mean()
                    loss = loss_cls - ENTROPY_ALPHA * gate_entropy

                    loss.backward()
                    optimizer.step()

            cs = client_model.state_dict()
            gate_update = {}
            keys_to_avg = [
                "gate_fc1_train.weight", "gate_fc1_train.bias",
                "gate_fc2_train.weight", "gate_fc2_train.bias",
                "gate_fc1_inf.weight",   "gate_fc1_inf.bias",
                "gate_fc2_inf.weight",   "gate_fc2_inf.bias",
            ]
            num_elements = 0
            for key in keys_to_avg:
                tensor = cs[key].clone()
                gate_update[key] = tensor
                num_elements += tensor.numel()
            bytes_sent = num_elements * 4
            print(f"  {client_name} uploads ~{bytes_sent/1024:.1f} KB for gate in this round")

            client_states[client_name] = gate_update

        new_global_state = copy.deepcopy(global_state)
        keys_to_avg = list(next(iter(client_states.values())).keys())
        num_clients = len(path_clients)
        for key in keys_to_avg:
            if is_bn_key(key):
                continue
            new_global_state[key] = sum(client_states[c][key] for c in path_clients.keys()) / num_clients
        global_model.load_state_dict(new_global_state)

        acc = evaluate_model_gpu(global_model, test_loader, device)
        print(f"[Gate-only] fine-tune round {rnd+1} accuracy: {acc:.2f}%")

    return global_model

def class_to_path_target(labels, device):
    """
    Build target gate distributions per sample:
    0–2 → [1,0,0] (Path1)
    3–5 → [0,1,0] (Path2)
    6–9 → [0,0,1] (Path3)
    """
    batch_size = labels.size(0)
    targets = torch.zeros(batch_size, 3, device=device)
    for i, y in enumerate(labels):
        y = y.item()
        if 0 <= y <= 2:
            targets[i, 0] = 1.0  # Path1
        elif 3 <= y <= 5:
            targets[i, 1] = 1.0  # Path2
        else:  # 6–9
            targets[i, 2] = 1.0  # Path3
    return targets

def print_param_stats(model, phase_name):

    return
    # Overall stats
    total_params = 0
    trainable_params = 0

    # Grouped stats
    groups = {
        'conv':   {'train': 0, 'frozen': 0},
        'gate':   {'train': 0, 'frozen': 0},
        'fc':     {'train': 0, 'frozen': 0},
        'other':  {'train': 0, 'frozen': 0}
    }
    
    for name, p in model.named_parameters():
        num = p.numel()
        total_params += num
        is_trainable = p.requires_grad

        # group
        if name.startswith('conv'):
            g = 'conv'
        elif name.startswith('gate_fc'):
            g = 'gate'
        elif name.startswith('fc'):
            g = 'fc'
        else:
            g = 'other'
        
        if is_trainable:
            trainable_params += num
            groups[g]['train'] += num
        else:
            groups[g]['frozen'] += num

    frozen_params = total_params - trainable_params
    ratio = trainable_params / total_params if total_params > 0 else 0.0

    print(f"\n[PARAM STATS] {phase_name}")
    print(f"  Total params:      {total_params}")
    print(f"  Trainable params:  {trainable_params}")
    print(f"  Frozen params:     {frozen_params}")
    print(f"  Trainable ratio:   {ratio*100:.2f}%")

    print(f"\n[DETAILED PARAM STATS] {phase_name}")
    for g in ['conv', 'gate', 'fc', 'other']:
        train = groups[g]['train']
        frozen = groups[g]['frozen']
        total = train + frozen
        if total == 0:
            continue
        gratio = train / total * 100.0
        print(f"  {g.upper()}:")
        print(f"    Trainable params: {train} ({gratio:.2f}% of {total})")
        print(f"    Frozen params:    {frozen}")

def server_pretrain_gate(model, train_loader, device, num_epochs=5, lam_gate=0.5):
    model.to(device)
    model.train()
    
    # 1. Configuration
    set_gate_trainable(model, True)  
    set_conv_trainable(model, False) 
    set_head_trainable(model, True)  # Head must be trainable to evaluate the paths
    freeze_bn_affine(model)

    # 2. Setup Loss Functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_gate = nn.CrossEntropyLoss() # Path routing loss

    gate_params = [p for n, p in model.named_parameters() if 'gate' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if n.startswith('fc') and p.requires_grad]

    # 2. Collect any OTHER trainable parameters (to avoid overlap)
    other_params = [
        p for n, p in model.named_parameters() 
        if p.requires_grad 
        and 'gate' not in n 
        and not n.startswith('fc')
    ]

    print(f"Gate params: {len(gate_params)} | Head params: {len(head_params)} | Others: {len(other_params)}")

    # 3. Initialize the optimizer with distinct groups
    optimizer = torch.optim.Adam([
        {'params': gate_params,  'lr': 1e-2}, # Higher learning rate for gate
        {'params': head_params,  'lr': 1e-3}, # Standard learning rate for head
        {'params': other_params, 'lr': 1e-3}  # Standard learning rate for anything else
    ])

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # 3. Get Targets: Convert labels to Path Indices (0, 1, or 2)
            # 0-2 -> Path 0 | 3-5 -> Path 1 | 6-9 -> Path 2
            path_targets = torch.zeros_like(y_batch)
            path_targets[(y_batch >= 3) & (y_batch <= 5)] = 1
            path_targets[(y_batch >= 6)] = 2

            # 4. Forward Pass
            # Ensure your model.forward returns the RAW logits for the gate, 
            # not the Softmax probabilities, for CrossEntropyLoss to work.
            logits, gate_logits = model(X_batch, y_batch) 
            
            loss_cls = criterion_cls(logits, y_batch)
            loss_gate = criterion_gate(gate_logits, path_targets)

            # 5. Combined Loss
            loss = loss_cls + lam_gate * loss_gate
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} Gate Pretrain Loss: {running_loss/len(train_loader):.4f}")


# MAIN EXECUTION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# strict_cnn = StrictGatedCNN(device).to(device)
strict_cnn = StrictGatedCNN(device=device, num_classes=10, mean=mean, std=std).to(device)

from torchinfo import summary  # pip install torchinfo
summary(strict_cnn, input_size=(BATCH_SIZE, 3, 64, 64))
# print(strict_cnn)


# print("\n=== CREATE DATA SHARD CLIENTS (NO LOCAL PRETRAINING) ===")
# num_clients = 9
# path_clients = create_data_shard_clients(strict_cnn, device, num_clients=num_clients)

# # No local training here; models in path_clients and strict_cnn stay at random init.
# # You can keep path_accuracies dict but it will not be used for per-client results.
# path_accuracies = {}


print("\n=== SERVER-SIDE GATE PRETRAINING ===")
server_pretrain_gate(strict_cnn, train_loader, device, num_epochs=30, lam_gate=0.1)

print("\n=== STRICT PATH FEDERATED LEARNING (DATA SHARDS) ===")
num_clients = 9
path_clients = create_data_shard_clients(strict_cnn, device, num_clients=num_clients)

# print("\n=== STRICT PATH FEDERATED LEARNING (FULL DATA PER CLIENT) ===")
# num_clients = 9
# path_clients = create_full_data_clients(strict_cnn, device, num_clients=num_clients)

from collections import Counter

def print_client_class_stats(path_clients, train_dataset):
    # all_labels is the labels tensor from the *original* train dataset
    all_labels = train_dataset.tensors[1].cpu().numpy()

    for client_name, client_info in path_clients.items():
        ds = client_info['loader'].dataset

        # If ds is a Subset(train), use its indices; otherwise assume it's the full train dataset
        if hasattr(ds, "indices"):          # torch.utils.data.Subset
            indices = ds.indices
        else:
            indices = np.arange(len(train_dataset))

        labels = all_labels[indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(
            f"{client_name}: {len(indices)} samples, "
            f"labels ~ (classes {unique}, counts {counts})"
        )

print("\n=== CLIENT CLASS STATS ===")
print_client_class_stats(path_clients, train)

path_accuracies = {}
NUM_LOCAL_EPOCHS = 3
CLIENT_LR = 0.0005
LAMBDA_GATE = 0.1   # strength of class→path gate regularization
# LAMBDA_GATE = 0.05  # strength of class→path gate regularization

for client_name, client_info in path_clients.items():
    model = client_info['model'].to(device)
    loader = client_info['loader']
    print(f"Training {client_name} on its local shard (with gate reg)...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CLIENT_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    WARMUP_EPOCHS = 3  # put this near CLIENT_LR / NUM_LOCAL_EPOCHS

    for epoch in range(NUM_LOCAL_EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits, gate_weights = model(X_batch, y_batch)
            loss_cls = criterion(logits, y_batch)

            # Optional: encourage non-collapsed gate (small entropy bonus)
            target_paths = class_to_path_target(y_batch, device)   # (B,3)
            loss_gate = F.mse_loss(gate_weights, target_paths)

            loss = loss_cls + LAMBDA_GATE * loss_gate
            loss.backward()
            optimizer.step()

        val_acc = evaluate_model_gpu(model, val_loader, device)
        print(f"{client_name} epoch {epoch+1}/{NUM_LOCAL_EPOCHS} val accuracy: {val_acc:.2f}%")
    
    # keep test only for final per-client number
    path_accuracies[client_name] = evaluate_model_gpu(model, test_loader, device)
    print(f"{client_name} final test accuracy: {path_accuracies[client_name]:.2f}%")





# After local training of all 9 clients
print("\n=== BUILD INITIAL GLOBAL MODEL BY FEDAVG OVER TRAINED CLIENTS ===")
# 1) Collect state_dicts from all clients
client_states = [client_info['model'].state_dict()
                 for client_info in path_clients.values()]

# 2) Initialize global_state as a deep copy of the first client
global_state = copy.deepcopy(client_states[0])

# 3) FedAvg: simple (unweighted) average over all keys
num_clients = len(client_states)
for k in global_state.keys():
    # stack tensors and take mean
    if is_personalized(k):
        continue
    stacked = torch.stack([cs[k].float() for cs in client_states], dim=0)
    global_state[k] = stacked.mean(dim=0)

# 4) Create global_model and load averaged weights
global_model = StrictGatedCNN(device=device, num_classes=10, mean=mean, std=std).to(device)
global_model.load_state_dict(global_state)

acc_after_initial_globale_model = evaluate_model_gpu(global_model, test_loader, device)
print(f"Accuracy after INITIAL GLOBAL MODEL: {acc_after_initial_globale_model:.2f}%")



# path_clients_temp = copy.deepcopy(path_clients)
# global_model_temp = copy.deepcopy(global_model)

path_clients_temp = path_clients
global_model_temp = global_model

print("\n=== FEDERATED GATE-ONLY FINE-TUNING (PRE-EXPERT) ===")
global_model_temp = federated_gate_finetune(path_clients_temp, global_model_temp, device, num_rounds=2)
acc_after_gate = evaluate_model_gpu(global_model_temp, test_loader, device)
print(f"Accuracy after Gate-only pre-fine-tune: {acc_after_gate:.2f}%")


print("\n=== FEDERATED EXPERT ROTATION (STEP 3) ===")
# global_model = copy.deepcopy(strict_cnn).to(device)
global_model = federated_expert_rotation(path_clients, global_model, device, num_rounds=10)

# normal global eval (uses gate)
acc_after_experts = evaluate_model_gpu(global_model, test_loader, device)
print(f"Accuracy after expert rotation: {acc_after_experts:.2f}%")

# NEW: eval with hard label→path routing (no gate)
acc_label_paths = evaluate_global_by_label_paths(global_model, test_loader, device)
print(f"Global accuracy with hard label→path routing: {acc_label_paths:.2f}%")

print("\n=== PER-EXPERT EVAL AFTER EXPERT FL ===")
per_expert_acc = evaluate_global_per_expert(global_model, test_loader, device)
for p, a in per_expert_acc.items():
    print(f"{p}: {a:.2f}%")

print("\n=== FEDERATED FC FINE-TUNING (STEP 4) ===")
global_model = federated_head_finetune(path_clients, global_model, device, num_rounds=1)
final_acc = evaluate_model_gpu(global_model, test_loader, device)
print(f"Final global accuracy (after head fine-tune): {final_acc:.2f}%")

analyze_final_gating(global_model, test_loader, device)

print("\n=== COMPLETE FL RESULTS ===")
for client_name, acc in path_accuracies.items():
    print(f"{client_name}: {acc:.2f}%")

print(f"Accuracy after INITIAL GLOBAL MODEL: {acc_after_initial_globale_model:.2f}%")
# print(f"Accuracy after Gate-only pre-fine-tune: {acc_after_gate:.2f}%")
print(f"After expert rotation: {acc_after_experts:.2f}%")
print(f"Final global (after gate fine-tune): {final_acc:.2f}%")



