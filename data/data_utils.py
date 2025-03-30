import os
import h5py
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

def mouse_x_one_hot_to_value(one_hot_vector):
    MOUSE_X = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
    index = int(np.argmax(one_hot_vector))
    return MOUSE_X[index]


def mouse_y_one_hot_to_value(one_hot_vector):
    MOUSE_Y = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    index = int(np.argmax(one_hot_vector))
    return MOUSE_Y[index]


class CSDataset(Dataset):
    def __init__(self, episode_ids, hdf5_dir, transform=None):
        super().__init__()
        self.episode_ids = episode_ids
        self.hdf5_dir = hdf5_dir
        self.transform = transform

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        hdf5_file_path = os.path.join(self.hdf5_dir, f'hdf5_dm_july2021_expert_{episode_id}.hdf5')
        with h5py.File(hdf5_file_path, 'r') as f:
            # Assume that each episode has 1000 frames (16 fps)
            # 1.5 seconds images -> 0.5 seconds actions
            # 24 frames images -> 8 frames actions
            n_frames = 32
            frame_start = random.randint(0, 1000 - 32)
            frame_end = frame_start + n_frames
            
            images = []
            actions = []
            for i in range(frame_start, frame_end):
                image_key = f'frame_{i}_x'
                action_key = f'frame_{i}_y'

                image = f[image_key][:] # [150, 280, 3(BGR)]
                image = image[..., ::-1] # [150, 280, 3(RGB)]
                action = f[action_key][:] # [51]
                processed_action = action[:7] # w,s,a,d,space,ctrl,shift
                processed_action = np.append(processed_action, action[10]) # r
                processed_action = np.append(processed_action, action[11:13]) # left/right click
                processed_action = np.append(processed_action, mouse_x_one_hot_to_value(action[13:36])) # mouse_x
                processed_action = np.append(processed_action, mouse_y_one_hot_to_value(action[36:50])) # mouse_y
                
                if self.transform:
                    image = Image.fromarray(image.astype(np.uint8))
                    image = self.transform(image)
                images.append(image)
                actions.append(processed_action)

            images = np.stack(images, axis=0) # [32, H, W, C]
            actions = np.stack(actions, axis=0) # [32, 12]
            
            inputs = {
                'image': torch.tensor(images[:24] , dtype=torch.float32),
            }
            
            targets = {
                'action': torch.tensor(actions[24:], dtype=torch.float32)
            }
            return inputs, targets


if __name__ == '__main__':
    hdf5_dir = '/home/wentao/cs_dateset/'  # replace with your directory
    n_episode = 188
    train_val_split = 0.8
    shuffled_ids = np.random.permutation(n_episode)
    train_ids = shuffled_ids[:int(train_val_split*n_episode)]
    val_ids = shuffled_ids[int(train_val_split*n_episode):]

    # Visualize the dataset
    train_dataset = CSDataset(train_ids, hdf5_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        image_data = inputs['image'].numpy()
        n_batch, n_frame, _, _, _ = image_data.shape
        for n in range(n_batch):
            fig, ax = plt.subplots()
            for i in range(n_frame):
                image = image_data[n, i] / 255.0
                ax.imshow(image)
                ax.axis('off')
                plt.pause(1.0 / 16)
            plt.close(fig)
        break
    
    # Use transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),             # Only height and width
        transforms.ToTensor(),                     # Converts [H, W, C] to [C, H, W] and scales to [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    train_dataset = CSDataset(train_ids, hdf5_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        image_data = inputs['image'].numpy()
        print("Image Shape After Transform: ", image_data.shape)
        break
    
    

