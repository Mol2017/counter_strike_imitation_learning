import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from data_utils import mouse_x_one_hot_to_value, mouse_y_one_hot_to_value


def calculate_key_press_rate(is_pressed):
    return np.mean(is_pressed, axis=-1)


# Set the directory path where your HDF5 files are located
hdf5_dir = '/home/wentao/cs_dateset/'  # replace with your directory
hdf5_files = glob.glob(os.path.join(hdf5_dir, '*.hdf5'))

success_count = 0
fail_count = 0
actions = []
# Inspect the input & output shape
for file in hdf5_files:
    try:
        with h5py.File(file, 'r') as f:
            print(f"Processing file: {file}")
            for i in range(1000):
                image_key = 'frame_{}_x'.format(i)
                action_key = 'frame_{}_y'.format(i)
                if image_key not in f or action_key not in f:
                    break

                # image 'frame_i_x': [150, 280, 3]
                # images are collected at 16 frames per second
                image = f[image_key][:]
                # print("Image shape: ", image.shape)
                
                # # Plot images
                # plt.imshow(image)
                # plt.axis('off')
                # plt.show()

                # action 'frame_i_y': [51]
                # 51 = n_keys(11) + n_clicks(2) + n_mouse_x(23) + n_mouse_y(15)
                # keys = [w,s,a,d,space,ctrl,shift,1,2,3,r]
                # clicks = [left, right]
                # mouse_x = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
                # mouse_y = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
                action = f[action_key][:]
                # print("Action shape: ", action.shape)
                
                processed_action = action[:7] # w,s,a,d,space,ctrl,shift
                processed_action = np.append(processed_action, action[10]) # r
                processed_action = np.append(processed_action, action[11:13]) # left/right click
                processed_action = np.append(processed_action, mouse_x_one_hot_to_value(action[13:36])) # mouse_x
                processed_action = np.append(processed_action, mouse_y_one_hot_to_value(action[36:50])) # mouse_y
                # print("Processed action: ", processed_action)
                actions.append(processed_action)
            success_count += 1
    
    except (OSError, ValueError) as e:
        print(f" Failed to open file: {file}")
        print(f" Error: {e}")
        fail_count += 1
        continue  
        
print(f"Successfully processed: {success_count} files")
print(f"Failed to process: {fail_count} files")


# Analysis
actions = np.stack(actions, axis=0)  # [N, 12]
print("Total action shape:", actions.shape)

# Print the key press rate
print("Key W press rate: ", calculate_key_press_rate(actions[:, 0]))
print("Key A press rate: ", calculate_key_press_rate(actions[:, 1]))
print("Key S press rate: ", calculate_key_press_rate(actions[:, 2]))
print("Key D press rate: ", calculate_key_press_rate(actions[:, 3]))
print("Key Space press rate: ", calculate_key_press_rate(actions[:, 4]))
print("Key Ctrl press rate: ", calculate_key_press_rate(actions[:, 5]))
print("Key Shift press rate: ", calculate_key_press_rate(actions[:, 6]))
print("Key R press rate: ", calculate_key_press_rate(actions[:, 7]))
print("Left click press rate: ", calculate_key_press_rate(actions[:, 8]))
print("Right click press rate: ", calculate_key_press_rate(actions[:, 9]))

# Plot the histogram of mouse x/y movements
mouse_x = actions[:, 10]
mouse_y = actions[:, 11]

fig1 = plt.figure()
plt.hist(mouse_x, bins=50)
plt.title("Histogram of Mouse X Movement")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)

fig2 = plt.figure()
plt.hist(mouse_y, bins=50)
plt.title("Histogram of Mouse Y Movement")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()