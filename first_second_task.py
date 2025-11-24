import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ThermalJSONDataset(Dataset):
    def __init__(self, root_dir):
        """
        Expects per-person folder:
        
        personX/
          ├─ personX_labelled_first.txt    # lines: "<timestamp> <0|1>"
          └─ personX_data.json             # a JSON *list* of:
                [
                  [ 609647, [[…frame0…],[…frame1…],…,[…frame34…]] ],
                  [ 609791, [[…],…] ],
                  …
                ]
        """
        self.samples = []
        self.frames  = {}

        # 1) load labels + JSON for each person
        for person in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, person)
            if not os.path.isdir(pdir): 
                continue

            # load labels
            lblf = os.path.join(pdir, f"{person}_labelled_first.txt")
            with open(lblf, 'r') as f:
                for line in f:
                    ts, lbl = line.strip().split()
                    self.samples.append((person, ts, int(lbl)))

            # load JSON list and convert to dict
            jf = os.path.join(pdir, f"{person}_data.json")
            with open(jf, 'r') as f:
                data_list = json.load(f)
            # data_list is [[timestamp, frames], …]
            # convert to { "timestamp": frames, … }
            self.frames[person] = {
                str(item[0]): item[1]
                for item in data_list
            }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        person, ts, label = self.samples[idx]
        # frames is a nested list of shape (35, H, W)
        frames = np.array(self.frames[person][ts], dtype=np.float32)  
        # → tensor shape (35, H, W)
        t = torch.from_numpy(frames)

        # normalize from [0,255]→[0,1], then standardize to mean=0,std=1
        t = t / 255.0
        t = (t - 0.5) / 0.5

        return t, label

class ThermalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(35,  8, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2,2)

        # after pools: spatial 24→12→6
        self.fc1   = nn.Linear(16 * 6 * 6, 784)
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(784, 100)
        self.fc3   = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(model, loader, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        print(f"Epoch {ep:3d}/{epochs} — "
              f"Loss: {total_loss/len(loader):.4f} — "
              f"Acc: {100*correct/total:5.2f}%")

if __name__ == "__main__":
    ROOT = "lopocv_first"
    ds   = ThermalJSONDataset(ROOT)
    dl   = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    net  = ThermalCNN()
    train(net, dl, epochs=100)
