# Self-supervised-Denoising-for-Multiphase-Flow-Biofilm-Microtomography CAPSTONE Project 2024-2025)
This project is about a self-supervised autoencoder denoising model that aims to denoise micro-CT scans without the noisy-clean image pairs.

# Project Content 📒
This project includes a Jupyter Notebook (Noise2NoisePyTorch.ipynb) containing all the necessary tools for processing images, creating dataset files, training, and evaluating the model. It also includes a sandbox code to experiment with different noise injection experiments and visualize their outputs.

# Dependencies ⚙️
This code was developed using Python v3.9.21 and CUDA Toolkit v12.6 <br>
To successfully run the program following modules need to be installed:
- NetCDF4 (1.7.2)
- Numpy (2.0.2)
- Scipy (1.13.1)
- Pillow (11.2.1)
- PyTorch (2.7.0+cu126)
- Torchvision (0.22.0+cu126)
- Matplotlib (3.9.4)
- Tqdm (4.67.1)
- Tensorboard (2.19.0)
- Scikit-Image (0.25.2)
- Imageio (2.37.0)

## Pytorch Installation <img src="https://pytorch.org/wp-content/uploads/2024/10/logo.svg" width="100" height="20">

If you wish to use a GPU for training, you will need to install the GPU version of the PyTorch library along with the appropriate NVIDIA CUDA Toolkit.<br>

Download and install the CUDA Toolkit by visiting NVIDIA's website [<a href="https://developer.nvidia.com/cuda-toolkit"> CUDA Toolkit </a>]<br><br>
After installing the Toolkit, we are ready to install PyTorch using the following command:<br>
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
For more details visit PyTorch's official site [<a href="https://pytorch.org/"> Pytorch Official </a>]

## Package Installations 📦

To install all the remaining packages/modules, you can use pip:<br>
```
pip install netCDF4 numpy scipy pillow matplotlib tqdm tensorboard scikit-image imageio
```
Or by installing 'requirements.txt'
```
pip install -r requirements.txt
```
# How to run 📋📝

# Sample Results 📜
