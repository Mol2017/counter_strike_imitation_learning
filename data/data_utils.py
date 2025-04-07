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
        self.hdf5_dir = hdf5_dir
        self.episode_ids = episode_ids
        self.transform = transform

        self.history_len = 4
        self.future_len = 2
        self.stride = 4
        
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        index_map = []
        for episode_id in self.episode_ids:
            file_path = os.path.join(self.hdf5_dir, f'hdf5_aim_july2021_expert_{episode_id}.hdf5')
            with h5py.File(file_path, 'r') as f:
                total_frames = 1000 # Each episode has 1000 frames (16 fps)
                for start in range(0, total_frames-(self.history_len+self.future_len), self.stride):
                    h_start = start
                    h_end = start + self.history_len
                    f_start = h_end
                    f_end = f_start + self.future_len
                    index_map.append((episode_id, h_start, h_end, f_start, f_end))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        episode_id, h_start, h_end, f_start, f_end = self.index_map[index]
        hdf5_file_path = os.path.join(self.hdf5_dir, f'hdf5_aim_july2021_expert_{episode_id}.hdf5')
        with h5py.File(hdf5_file_path, 'r') as f:
            images = []
            actions = []
            for i in range(h_start, f_end):
                image_key = f'frame_{i}_x'
                action_key = f'frame_{i}_y'

                image = f[image_key][:] # [150, 280, 3(BGR)]
                image = image[..., ::-1] # [150, 280, 3(RGB)]             
                if self.transform:
                    image = Image.fromarray(image.astype(np.uint8))
                    image = self.transform(image)
                
                action = f[action_key][:] # [51]
                processed_action = np.concatenate([
                    action[:7], # w,s,a,d,space,ctrl,shift
                    [action[10]], # r
                    action[11:13], # left/right click
                    [mouse_x_one_hot_to_value(action[13:36])], # mouse_x
                    [mouse_y_one_hot_to_value(action[36:50])], # mouse_y
                ])

                images.append(image)
                actions.append(processed_action)

            images = np.stack(images, axis=0) # [32, H, W, C]
            actions = np.stack(actions, axis=0) # [32, 12]
            
            inputs = {
                'image': torch.tensor(images[:self.history_len] , dtype=torch.float32),
            }
            
            targets = {
                'action': torch.tensor(actions[self.history_len:], dtype=torch.float32)
            }
            return inputs, targets


if __name__ == '__main__':
    hdf5_dir = '/home/wentao/cs_dataset_aim/'  # replace with your directory
    n_episode = 44
    train_val_split = 1
    shuffled_ids = np.random.permutation(n_episode)
    train_ids = shuffled_ids[:int(train_val_split*n_episode)]

    # Visualize the dataset
    dataset = CSDataset(shuffled_ids, hdf5_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
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
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        image_data = inputs['image'].numpy()
        action_data = targets['action'].numpy()
        print("Image Shape After Transform: ", image_data.shape)
        print("Action Shape: ", action_data.shape)
        break
    
    

