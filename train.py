import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.data_utils import CSDataset
from torch.utils.data import DataLoader

from utils.utils import move_dict_to_device
from model.cs_transformer_v1 import CSTransformerV1 as CSTransformer
import torchvision.transforms as transforms


# Hyperparameters
hdf5_dir = '/home/wentao/cs_dataset_aim/'  # replace with your directory
n_episode = 44 # 188 in total
train_val_split = 0.8
n_epoch = 400

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Only height and width
    transforms.ToTensor(),                     # Converts [H, W, C] to [C, H, W] and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Dataset
shuffled_ids = np.random.permutation(n_episode)
train_ids = shuffled_ids[:int(train_val_split*n_episode)]
val_ids = shuffled_ids[int(train_val_split*n_episode):]
train_dataset = CSDataset(train_ids, hdf5_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

val_dataset = CSDataset(train_ids, hdf5_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=1, prefetch_factor=1)

# Model 
model = CSTransformer().to(device)

# Loss function
criterion = nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(n_epoch):
    # ----- TRAINING -----
    model.train()
    train_loss = 0.0

    for inputs, targets in train_dataloader:
        inputs, targets = move_dict_to_device(inputs, device), move_dict_to_device(targets, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs['action'], targets['action'])

        # bce_loss = F.binary_cross_entropy(outputs['action'][..., :10], targets['action'][..., :10])
        # l2_loss = F.mse_loss(outputs['action'][..., 10:], targets['action'][..., 10:])
        # loss = bce_loss + l2_loss
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)

    # ----- VALIDATION -----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = move_dict_to_device(inputs, device), move_dict_to_device(targets, device)
            
            outputs = model(inputs)
            loss = criterion(outputs['action'], targets['action'])
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    print(f"[Epoch {epoch+1}/{n_epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "cstransformer1.pth")
