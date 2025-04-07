import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.data_utils import CSDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
inv_transform = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# Dataset
test_dataset = CSDataset(np.arange(n_episode), hdf5_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

# Model 
model = CSTransformer().to(device)
model.load_state_dict(torch.load('cstransformer.pth'))
model.eval()

# Loss function
criterion = nn.L1Loss()

for epoch in range(n_epoch):
    # ----- TEST -----
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = move_dict_to_device(inputs, device), move_dict_to_device(targets, device)
            
            outputs = model(inputs)
            loss = criterion(outputs['action'], targets['action'])
            test_loss = loss.item()
            
            # Plot & Print
            image_data = inputs['image'].cpu()
            action_data = targets['action'].cpu()
            pred_action_data = outputs['action'].cpu()
            n_batch, n_frame, _, _, _ = image_data.shape
            for n in range(n_batch):
                fig, ax = plt.subplots()
                for i in range(n_frame):
                    image = inv_transform(image_data[n, i])
                    image = image.numpy().transpose(1, 2, 0)
                    ax.imshow(image)
                    ax.axis('off')
                    plt.pause(1.0 / 16)
                print("Target Action: ", action_data)
                print("Predicted Action: ", pred_action_data)
                plt.show()
            break
        break
