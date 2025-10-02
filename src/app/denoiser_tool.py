import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
from pathlib import Path
core_dir = Path(__file__).resolve().parent.parent / 'core'
sys.path.append(str(core_dir))
from network import Noise2Noise
from utils import NoiseType

import streamlit as st
import torch
from typing import Dict, Tuple, List
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SNAPSHOTS = {}
MODEL = Noise2Noise()

@st.cache_resource
def load_snapshot(snapshot_name:str):
    snpsht = torch.load(SNAPSHOTS[snapshot_name], map_location='cpu')
    MODEL.load_state_dict(snpsht['STATE_DICT'])
    MODEL.eval()

def load_imput_image(input) -> np.ndarray:
    image = np.array(Image.open(input).convert('L'), dtype=np.float32)
    return image

def update_snapshot():
    name = st.session_state["snapshot_select"]
    if name is not None:
        load_snapshot(name)

def get_snapshots():
    snapshot_paths = Path("./snapshots").glob("*.pth")
    global SNAPSHOTS
    SNAPSHOTS = {path.name: path for path in snapshot_paths}
    return SNAPSHOTS.keys()

def preprocessing(input, **kwargs) -> np.ndarray:
    return input

def plot_hist(images:Dict[str, np.ndarray]) -> np.ndarray:
    plt.figure()
    
    if np.allclose(images['Input'], images['Noisy Image']):
        # 2 different plots
        del images['Noisy Image']
    
    for label, img in images.items():
        plt.hist(img.astype(np.uint8).ravel(), bins=256, range=(0, 255), alpha=0.4, density=True, histtype='bar', label=label)

    plt.title('Image Histograms')
    plt.xlabel('Pixel Values (0-255)')
    plt.ylabel('Frequency Density')
    plt.legend()
    plt.grid(True)

    plt.gcf().canvas.draw()
    w, h = plt.gcf().canvas.get_width_height()
    buf = np.frombuffer(plt.gcf().canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    buf = buf[:, :, [1, 2, 3, 0]]
    # buf = buf[:, :, :3]
    return buf

def try_eval(input, snap, **kwargs):
    if input is not None and snap is not None:
        output, histogram = eval(input, **kwargs)
        st.session_state["Output_image"] = output
        st.session_state["Output_hist"] = histogram

def eval(input:np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    global MODEL
    model = MODEL
    model.eval()
    output = None
    with torch.inference_mode():
        input_tensor = torch.from_numpy(input).div(255.0).sub(0.5).unsqueeze(0)
        # a = (input + 0.5).astype(np.uint8)
        # return a, a
        # Perform pre-processing on input.
        processed_ip = preprocessing(input_tensor, **kwargs)
        # Use Model
        output = model(processed_ip)
        
        # Calculate Loss if a clean image was used.
        # eval_loss = kwargs.get('loss_fn')(input/255.0, output)
        
        # Generate histograms
        histogram = plot_hist({'Input':input, 'Noisy Image':processed_ip.add(0.5).mul(255.0).numpy(), 'Output':output.squeeze(0).add(0.5).mul(255.0).numpy()})

        output = output.squeeze(0).add(0.5).mul(255).numpy().astype(np.uint8)
        histogram = histogram.astype(np.uint8)
    return output, histogram

def UI():
    # Parameters
    input = None
    get_snapshots()
    kwargs = {}
    
    # UI Title
    st.title("Noise2Noise Denoiser App")

    # 2 Columns - Left, Right
    ip_left, ip_right = st.columns(2)

    # Left
    selected_snapshot = ip_left.selectbox("Snapshot To Use:", SNAPSHOTS.keys(), index=None, placeholder="Select Snapshot before proceeding", key="snapshot_select", on_change=update_snapshot, width=300)
    # ip_left.markdown(SNAPSHOTS)
    ip_left.markdown(f"Snapshot Selected: :green[{selected_snapshot}]")

    # Right
    uploaded_file = ip_right.file_uploader("Upload the Input Image:", type=["png", "jpg", "jpeg"])
    # right.markdown(type(uploaded_file))
    if uploaded_file is not None:
        input = load_imput_image(uploaded_file)
        ip_right.image((input+0.5).astype(np.uint8), caption="Preview")

    # Central
    start_denoiser = st.button("Denoise", icon=":material/settings:", on_click=try_eval, args=(input, selected_snapshot), kwargs=kwargs, use_container_width=True)
    if start_denoiser:
        op_left, op_right = st.columns(2)
        op_left.image((input+0.5).astype(np.uint8), caption="Input", use_container_width=True)
        op_right.image(st.session_state['Output_image'], caption="Denoised Output", use_container_width=True)
    
        st.image(st.session_state['Output_hist'], caption="Pixel Frequency Density Distribution Histogram", use_container_width=True)

if __name__ == '__main__':
    UI()