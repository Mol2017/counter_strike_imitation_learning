import os
import time
import torch
import numpy as np
from cs_transformer import CSTransformerV1  # your PyTorch model
from screen_input import grab_window
from key_input import key_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import w_char, a_char, s_char, d_char, space_char, r_char
from key_output import left_click, release_left_click

# === Settings ===
IS_MOUSEMOVE = True
IS_WASD = True
IS_JUMP = True
IS_RELOAD = True
IS_CLICKS = True

N_FRAMES = 4
IMG_SIZE = (224, 224)
ACTION_DIM = 12

# === Init ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSTransformerV1().to(device).eval()

# Optional: load trained weights
# model.load_state_dict(torch.load('your_model.pth'))

# === Game window & screen capture setup ===
from win32gui import FindWindow, SetForegroundWindow
hwin = FindWindow(None, 'Counter-Strike: Global Offensive - Direct3d 9')
SetForegroundWindow(hwin)
time.sleep(1)

# Capture screen center for mouse control
from mss import mss
sct = mss()
monitor = sct.monitors[1]
Wd, Hd = monitor["width"], monitor["height"]
mouse_x_mid, mouse_y_mid = Wd // 2, Hd // 2

# === Buffers ===
recent_imgs = []

# === Main Loop ===
print("Starting agent loop...")
while True:
    loop_start = time.time()

    # 1. Capture frame
    img = grab_window(hwin, game_resolution=(Wd, Hd), resize=IMG_SIZE, SHOW_IMAGE=False)
    img = img.transpose(2, 0, 1)  # to [C, H, W]
    recent_imgs.append(img)
    if len(recent_imgs) < N_FRAMES:
        time.sleep(0.05)
        continue
    if len(recent_imgs) > N_FRAMES:
        recent_imgs.pop(0)

    # 2. Prepare input
    stack = np.stack(recent_imgs, axis=0)  # [T, C, H, W]
    x = torch.tensor(stack / 255.0, dtype=torch.float32).to(device).unsqueeze(0)  # [1, T, C, H, W]

    # 3. Run model
    with torch.no_grad():
        out = model({"image": x})
    actions = out['action'].squeeze(0).cpu().numpy()  # [2, 12]
    action = actions[0]  # Use the first predicted action

    # 4. Decode action
    keys_pred = action[:6]  # ['w','a','s','d','space','r']
    click_prob = action[6]  # left click
    mouse_x = action[7] * 300  # normalize back
    mouse_y = action[8] * 300

    # 5. Apply keyboard input
    keys = ['w','a','s','d','space','r']
    key_chars = [w_char, a_char, s_char, d_char, space_char, r_char]
    for i, (k, c) in enumerate(zip(keys, key_chars)):
        if keys_pred[i] > 0.5:
            HoldKey(c)
        else:
            ReleaseKey(c)

    # 6. Apply mouse
    if IS_MOUSEMOVE:
        set_pos(mouse_x_mid + int(mouse_x), mouse_y_mid + int(mouse_y), Wd, Hd)
    if IS_CLICKS:
        if click_prob > 0.5:
            left_click()
        else:
            release_left_click()

    # 7. Exit
    if 'Q' in key_check():
        print("Exiting agent loop...")
        break

    # 8. Wait for next frame
    while time.time() - loop_start < 1/16:
        time.sleep(0.001)

# Cleanup
for c in [w_char, a_char, s_char, d_char, space_char, r_char]:
    ReleaseKey(c)
release_left_click()
