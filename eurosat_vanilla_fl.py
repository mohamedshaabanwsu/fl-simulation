import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
BATCH_SIZE = 256
LR = 1e-4
LOCAL_EPOCHS = 3
FED_ROUNDS = 10
NUM_CLIENTS = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./input/6/EuroSAT"

# --- DATA LOADING ---
load_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class EuroSATDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.img_paths = [os.path.join(BASE_DIR, f) for f in df["Filename"]]
        self.labels = df["Label"].values
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img) if self.transform else load_transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# --- ORIGINAL BACKBONE MODEL (No Gating / No Masks) ---
class StrictCNN_NoGating(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 10 conv layers as defined in your original architecture
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
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8   = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9   = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10   = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head from original architecture
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass using your original pooling strategy
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool(x) # 32
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x) # 16
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool(x) # 8
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool(x) # 4
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool(x) # 2

        # 5 Non-pooling layers
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc3(x)

# --- FEDERATED UTILS ---
def federated_avg(global_model, client_weights):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights[i][k].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# --- MAIN ---
if __name__ == "__main__":
    train_set = EuroSATDataset(os.path.join(BASE_DIR, "train.csv"))
    test_loader = DataLoader(EuroSATDataset(os.path.join(BASE_DIR, "test.csv")), batch_size=BATCH_SIZE)

    indices = torch.randperm(len(train_set))
    shard_size = len(train_set) // NUM_CLIENTS
    client_indices = [indices[i*shard_size:(i+1)*shard_size] for i in range(NUM_CLIENTS)]

    global_model = StrictCNN_NoGating().to(DEVICE)

    for r in range(FED_ROUNDS):
        print(f"\n--- Round {r+1}/{FED_ROUNDS} ---")
        weights = []
        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(torch.utils.data.Subset(train_set, client_indices[i]), batch_size=BATCH_SIZE, shuffle=True)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
            
            local_model.train()
            for _ in range(LOCAL_EPOCHS):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    F.cross_entropy(local_model(x), y).backward()
                    optimizer.step()
            weights.append(local_model.state_dict())
            print(f"Client {i+1} done.")
            
        federated_avg(global_model, weights)
        print(f"Global Test Acc: {evaluate(global_model, test_loader):.2f}%")
