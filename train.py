import gc
import json
import time
import warnings
import numpy as np
import argparse

import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime as dt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler as lr_sch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
from torchvision.transforms.functional import gaussian_blur
from torch.utils.tensorboard import SummaryWriter

from dataset import CTScans
from network import Noise2Noise
from utils import Logger, NoiseType
import train_config

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 loss_fn: Callable,
                 gpu_id: int,
                 save_interval:int = 10
                 ) -> None:
        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval

    def _save_config(self, **kwargs):
        '''
        This function saves the parameters and hyperparameters used for the current run of the experiment to 'run-config.txt'.
        '''
        logger = self.Logger(str(self.results_dir / "run-config.txt"))
        # kwargs = {**self.init_kwargs, **kwargs}
        config = {
            'loss': str(self.loss_fn),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'post_op': self.post_op,
            'max_epochs': self.max_epochs,
            'start_epoch': self.start_epoch,
            'results_dir': str(self.results_dir),
            'corrupt_targets': kwargs.get("n2n", True),
            'batch_size': self.batch_size,
            'checkpoint_save_interval': self.save_interval,
            'augmentation_params': kwargs.get("augment_params", dict()),
            'corruption_type': kwargs.get("corruption_type", None),
            'noisy_target_factor': self.target_noise_injection_factor, 
            'corruption_params': kwargs.get("corruption_params", dict())
        }
        
        logger.start()
        print("############ Config #############")
        print(json.dumps({'config': config}, indent=4))
        print("#################################")
        logger.stop()

    # Augment Data
    def _augment_data(self, imgs:torch.Tensor, specs:torch.Tensor, augment_params:Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function performs augmentation operations like translation.
        
        Args:
            imgs (Tensor: float): Tensor of images to augment.
            specs (Tensor: complex): Tensor of spectral representations of the images.
            augment_params (Dict: [str, int]): Dictionary of augmentation parameters.
        
        Returns:
            augmented_images (Tensor: float): Tensor of the augmented images.
            augmented_spectrals (Tensor: complex): Tensor of the spectral representations of the augmented images.
        '''
        t = augment_params.get('translate', 0)
        if t <= 0:
            return imgs, specs
        else:
            trans = torch.randint(-t, t + 1, size=(2, ))
            
            cache_key = (int(trans[0].item()), int(trans[1].item()))

            if cache_key not in self.augment_translate_cache:
                # Create delta image with a shifted impulse
                x = torch.zeros(imgs[0].shape, dtype=torch.float32)
                y_idx = (trans[0].item() + imgs[0].shape[0]) % imgs[0].shape[0]
                x_idx = (trans[1].item() + imgs[0].shape[1]) % imgs[0].shape[1]
                x[y_idx, x_idx] = 1.0
                kernel_spec = self._compute_spectral(x, shift=True)
                self.augment_translate_cache[cache_key] = kernel_spec
                kernel_spec = kernel_spec.to(self.device)
            else:
                kernel_spec = self.augment_translate_cache[cache_key].to(self.device)

            new_imgs = torch.roll(imgs, shifts=(cache_key[0], cache_key[1]), dims=(1, 2))
            new_specs = specs * kernel_spec[None, :, :]
            return new_imgs, new_specs

    # Compute Spectral
    def _compute_spectral(self, imgs:torch.Tensor, shift:bool=False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the spectral image using the 2D Fourier Transform of spatial images.
        
        Args:
            imgs (Tensor): Image to convert to spectral representation.
            shift (bool): Boolean to perform fast fourier shift or not.
            **kwargs:
                ifftshift (bool): Boolean for if it's an inverse fast fourier shift. _Default: False_
                magnitude (bool): Boolean to get log of magnitude of the spectral representation. _Default: False_
                scaled (bool): Boolean to perform logarithmic scaling of the magnitude. _Default: False_
                normalize (bool): Boolean to normalize the magnitude to the range [0, 1] _Default: False_
            
        Returns:
            spectral_image (Tensor): The spectral representation.
            magnitude (Tensor: float)[Optional]: Log of the magnitude of the spectral representation.
        """
        magnitude = kwargs.get("magnitude", False)
        scaled = kwargs.get("scaled", False)
        normalize = kwargs.get("normalize", False)
        ifftshift = kwargs.get("ifftshift", False)

        specs = torch.fft.fft2(imgs)
        if shift:
            if ifftshift:
                output_shifted_specs = torch.fft.ifftshift(specs, dim=(-2, -1))
            else:
                output_shifted_specs = torch.fft.fftshift(specs, dim=(-2, -1))
            output_specs = output_shifted_specs
        else:
            output_specs = specs

        if magnitude:
            # Get the magnitude of the spectral representation.
            mag = torch.abs(output_specs)
            if scaled:
                # Scale the values with log1p.
                mag = torch.log1p(mag)
            if normalize:
                # Normalize to range [0, 1]
                mag_min = mag.amin(dim=(-2, -1), keepdim=True)
                diff = mag.amax(dim=(-2, -1), keepdim=True) - mag_min
                if diff.min() > 0:
                    mag = (mag - mag_min) / diff 
                else:
                    mag = torch.zeros_like(mag)
            
            return output_specs.to(torch.complex64), mag
        
        return output_specs.to(torch.complex64)
    
    # Compute Spatial
    def _compute_spatial(self, specs:torch.Tensor, shift:bool=False, **kwargs) -> torch.Tensor:
        """
        Computes the spatial image using the spectral representations.
        
        Args:
            specs (Tensor): Spectral representation to convert to spatial image.
            shift (bool): Boolean to perform fast fourier shift or not.
            **kwargs:
                ifft (bool): Boolean for if it's an inverse fast fourier shift. _Default: True_
            
        Returns:
            spectral_image (Tensor): The spectral representation.
        """
        ifftshift = kwargs.get("ifftshift", True)

        if shift:
            if ifftshift:
                shifted_specs = torch.fft.fftshift(specs, dim=(-2, -1))
            else:
                shifted_specs = torch.fft.ifftshift(specs, dim=(-2, -1))
            output_imgs = torch.fft.ifft2(shifted_specs).real.to(torch.float32)
        else:
            output_imgs = torch.fft.ifft2(specs).real.to(torch.float32)
        
        return output_imgs
    
    # Inject Noise
    corruption_masks={}
    def _inject_noise(self, imgs:torch.Tensor, specs:torch.Tensor, **kwargs):
        '''
            Injects images with artificial noise values.
            
            Args:
                images (Tensor): Source image tensor
                specs (Tensor): Spectral representation of the input images.
                **kwargs: keyword arguments for respective corruption type.
            
            Returns:
                corrupted_images (Tensor): Images after adding the noise values.
                corrupted_spectrals (Tensor): The spectral representation of noisy images.
                corruption_mask (Tensor: float): The mask used to inject noise.
        '''
        _, H, W = imgs.shape
        unique_key_freq = 1
        corruption_type = kwargs.get("corruption_type")
        corruption_params = kwargs.get("corruption_params")
        noise_injection_factor = corruption_params.get("noise_injection_factor", 1.0)

        output_imgs, output_specs, output_mask = imgs, specs, torch.ones_like(imgs)
        # Choose to corrupt or not, and choose the corruption type.
        if corruption_type == NoiseType.NO_NOISE:
            output_imgs, output_specs, output_mask = imgs, specs, torch.ones_like(imgs)

        elif corruption_type == NoiseType.GAUSSIAN:
            gaussian_params = corruption_params.get("gaussian_params", dict())
            if "std" not in gaussian_params:
                warnings.warn("Missing Parameter Warning: Using default values for - gaussian_params:{\"Gaussian Standard Deviation\" = 0.1}")
            std = corruption_params.get("std", 0.1) * noise_injection_factor

            cache_key = (self.epoch%unique_key_freq, "gaussian", std)
            if cache_key not in self.corruption_masks:
                mask = torch.normal(mean=0.0, std=std, size=imgs[0].shape, device=self.device)
                
            output_imgs = imgs + mask
            output_specs = self._compute_spectral(output_imgs)
            
            mask = torch.log1p(self._compute_spectral(mask).abs())
            output_mask = (mask - mask.min())/(mask.max() - mask.min())
            output_mask *= -1
            
        elif corruption_type == NoiseType.POISSON:
            poisson_params = corruption_params.get("poisson_params", dict())
            if "poisson_strength" not in poisson_params:
                warnings.warn("Missing Parameter Warning: Using default values for - poisson_params:{\"poisson_strength\" = 0.102}")
            corruption_strength = poisson_params.get("poisson_strength", 0.102) * noise_injection_factor
            
            if "distribution" not in poisson_params:
                warnings.warn("Missing Parameter Warning: Using default values for - poisson_params:{\"distribution\" = \"uniform\"}")
            distribution = poisson_params.get("distribution", "uniform")

            mask = None
            cache_key = (self.epoch%unique_key_freq, "poisson", corruption_strength, distribution)
            if cache_key not in self.corruption_masks:
                # Create a corruption mask
                if distribution == "uniform":
                    if "mask_ratio" not in kwargs["corruption_params"]:
                        warnings.warn("Missing Parameter Warning: Using default values for - corruption_params:{\"mask_ratio\" = 0.05}")
                    criterion = kwargs["corruption_params"].get("mask_ratio", 0.05)
                    
                elif distribution == "gaussian":
                    if "sigma" not in kwargs["corruption_params"]:
                        raise ValueError("Missing 'sigma' in corruption_params.")
                        
                    sigma = kwargs["corruption_params"].get("poisson_sigma", 1.0)
                    y = torch.arange(H, dtype=torch.float32, device=device) - H // 2
                    x = torch.arange(W, dtype=torch.float32, device=device) - W // 2
                    yy, xx = torch.meshgrid(y, x, indexing="ij")
                    criterion = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                    
                else:
                    # Default to uniform distribution with mask_ratio = 0.05
                    criterion = 0.05
                    
                mask = (torch.rand((H, W), device=self.device) < criterion).float()
                self.corruption_masks[cache_key] = mask
            else:
                mask = self.corruption_masks[cache_key]
            
            poisson_noise = torch.poisson(torch.abs(imgs + 0.5) * corruption_strength) / corruption_strength - 0.5
            batch_mask = mask.unsqueeze(0)
            output_imgs = torch.where(batch_mask.bool(), poisson_noise, imgs)
            output_specs = self._compute_spectral(output_imgs)
            
            mask = torch.log1p(self._compute_spectral(mask).abs())
            output_mask = (mask - mask.min())/(mask.max() - mask.min())
            output_mask *= -1
        
        elif corruption_type == NoiseType.BERNOULLI:
            if "p_edge" not in corruption_params:
                    warnings.warn("Missing Parameter Warning: Using default values for - corruption_params:{\"p_edge\" = 0.025}")

            p_edge = corruption_params.get("p_edge", 0.025) * noise_injection_factor
            cache_key = (self.epoch%unique_key_freq, "bernoulli", p_edge)
            if cache_key not in self.corruption_masks:
                # Create a corruption mask
                y = torch.arange(H, dtype=torch.float32, device=self.device) - H // 2
                x = torch.arange(W, dtype=torch.float32, device=self.device) - W // 2
                yy, xx = torch.meshgrid(y**2, x**2, indexing='ij')
                r_dist = torch.sqrt(xx + yy)
                prob_mask = (p_edge ** (2.0 / W)) ** r_dist
                
                keep = (torch.rand(size=(H, W), device=self.device, dtype=torch.float32) ** 2) < prob_mask
                keep = keep & torch.flip(keep, dims=[0, 1])
                
                self.corruption_masks[cache_key] = (keep, prob_mask)
                
            else:    
                keep, prob_mask = self.corruption_masks[cache_key]
            
            # Apply Mask
            mskd_specs = specs * keep
            output_mask = keep.to(torch.float32)
            output_specs = self._compute_spectral(mskd_specs / torch.where(keep, prob_mask, 1e-8), ifft=True)
            output_imgs = torch.fft.ifft2(output_specs).real.float()
            
            output_mask *= -1

        elif corruption_type == NoiseType.GAUSSIAN_BLUR:
            blur_params = corruption_params.get("blur_params", dict())
            if "blur_kernel" not in blur_params:
                warnings.warn("Missing Parameter Warning: Using default values for - blur_params:{\"Blur Kernel\" = 15}")
            kernel = blur_params.get("blur_kernel", 15)
            
            if "blur_sigma" not in blur_params:
                warnings.warn("Missing Parameter Warning: Using default values for - blur_params:{\"Blur Sigma\" = 1.0}")
            sigma = blur_params.get("blur_sigma", 1.0) * noise_injection_factor
            
            output_imgs = gaussian_blur(img=imgs, kernel_size=kernel, sigma=sigma)
            output_specs = self._compute_spectral(output_imgs)
            output_mask = torch.zeros_like(output_imgs)
        
        else:
            raise ValueError(f"Requested \"{corruption_type}\" which is an invalid corruption type!")
        
        return output_imgs, output_specs, output_mask
    
    # Noise Injector
    def _noise_injector(self, imgs:torch.Tensor, specs:torch.Tensor, corruption_types:List[NoiseType], corruption_params:Dict):
        '''
            This function sequentially injects noise to a tensor of images.

            Args:
                imgs (Tensor): Images to add noise to.
                specs (Tensor): Spectral representation of the images.
                corruption_types (List[str]): List of corruptions to add in sequence.
                corruption_params (Dict): Parameters for respective corruption type.
            
            Returns:
                corrupted_images (Tensor): Corrupted images.
                corrupted_spectrals (Tensor): The spectral representation.
                corruption_mask (Tensor): The effective corruption mask for the sequence of noise patterns.
        '''
        output_imgs, output_specs, output_masks = imgs, specs, torch.ones_like(imgs)
        if corruption_types:
            for corruption in corruption_types:
                output_imgs, output_specs, masks = self._inject_noise(output_imgs, output_specs, corruption_type=corruption, corruption_params=corruption_params)
                output_masks += masks
            
        return output_imgs, output_specs, output_masks

    # Post-op
    def _post_op(self, op_type:str, input:torch.Tensor, spec_value:torch.Tensor, spec_mask:torch.Tensor) -> torch.Tensor:
        '''
        Performs the post-operation procedure of forcing known frequencies before the training.

        Args:
            op_type (str): Type of post-op requested.
            denoised (Tensor: float): The image to perform post-op on.
            spec_value (Tensor: complex): The spectral representation of the image.
            spec_mask (Tensor: float): The mask used for post-op in spectral domain.
            
        Returns:
            denoised (Tensor: float): The image after post-op is performed.
        '''
        if op_type =='fspec':
            def fftshift3d(x, ifft=False):
                '''
                Performs a mannual Fast Fourier Shift over a 3D tensor.
                '''
                assert len(x.shape) == 3
                s0 = (x.shape[-2] // 2) + (0 if ifft else 1)
                s1 = (x.shape[-1] // 2) + (0 if ifft else 1)
                x = torch.cat([x[:, s0:, :], x[:, :s0, :]], dim=1)
                x = torch.cat([x[:, :, s1:], x[:, :, :s1]], dim=2)
                return x
            
            # print("Force denoised spectrum to known values.")
            # FFT
            denoised_spec = torch.fft.fft2(input)     
            # FFT shift
            denoised_spec = fftshift3d(denoised_spec, ifft=False)
            # Ensure correct dtypes and device
            spec_value = spec_value.to(denoised_spec.dtype).to(denoised_spec.device)
            spec_mask = spec_mask.to(denoised_spec.dtype).to(denoised_spec.device)
            # Force known frequencies using mask
            denoised_spec = spec_value * spec_mask + denoised_spec * (1. - spec_mask)
            # Shift back and IFFT
            denoised = torch.fft.ifft2(fftshift3d(denoised_spec, ifft=True)).real
        else:
            denoised = input
            warnings.warn("Invalid Post-Op requested. No Post-Op performed.")
        return denoised
    
    def _post_op2(self, op_type:str, input:torch.Tensor, spec_value:torch.Tensor, spec_mask:torch.Tensor) -> torch.Tensor:
        '''
        Performs the post-operation procedure of forcing known frequencies before the training.

        Args:
            op_type (str): Type of post-op requested.
            input (Tensor: float): The image to perform post-op on.
            spec_value (Tensor: complex): The spectral representation of the image.
            spec_mask (Tensor: float): The mask used for post-op in spectral domain.
            
        Returns:
            denoised (Tensor: float): The image after post-op is performed.
        '''
        if op_type =='fspec':
            # print("Force denoised spectrum to known values.")  
            # FFT shift
            denoised_spec = self._compute_spectral(input, shift=True)
            # Ensure correct dtypes and device
            spec_value = spec_value.to(denoised_spec.dtype).to(denoised_spec.device)
            spec_mask = spec_mask.to(denoised_spec.dtype).to(denoised_spec.device)
            # Force known frequencies using mask
            denoised_spec = spec_value * spec_mask + denoised_spec * (1. - spec_mask)
            # Shift back and IFFT
            denoised = torch.fft.ifft2(self._compute_spectral(denoised_spec, shift=True, ifftshift=True)).real.to(torch.float32)
        else:
            denoised = input
            warnings.warn("Invalid Post-Op requested. No Post-Op performed.")

        return denoised

    # Preprocess Batch
    def _preprocessbatch(self, imgs:torch.Tensor, specs:torch.Tensor, noisy_targets:bool=True, **kwargs):
        '''
        This function performs preprocessing on a batch of images.

        Args:
            imgs (Tensor): Batch of images to process.
            specs (Tensor): Spectral representation of images.
            noisy_targets (bool): Boolean to select noisy or clean targets.
            inject_noise (bool): Boolean to add noise to images or not.
            **kwargs:
                corruption_types (List[str]): List of corruptions to be injected in sequence.
                corruption_params (Dict): Parameters for the respective corruption type.
        
        Returns:
            processed_images (Tensor): Images after processing.
            processed_targets (Tensor): Targets for training.
            processed_spectrals (Tensor): The spectral representation.
            corruption_mask (Tensor: float): The effective mask used for injecting noise values.
            original_images (Tensor): Original images before any preprocessing.
        '''
        inject_noise = kwargs.get("inject_noise", True)
        corruption_types = kwargs.get("corruption_types", [NoiseType.NO_NOISE])
        corruption_params = kwargs.get("corruption_params")
            
        # Augment Images
        augmented_imgs, augmented_specs = self._augment_data(imgs, specs, augment_params=kwargs.get("augment_params", dict()))

        # Inject Noise
        if inject_noise:
            processed_images, processed_spectrals, corruption_mask = self._noise_injector(augmented_imgs, augmented_specs, corruption_types=corruption_types, corruption_params=corruption_params)
        else:
            processed_images, processed_spectrals, corruption_mask = augmented_imgs, augmented_specs, torch.ones_like(augmented_imgs)

        # Corrupt Targets
        if noisy_targets:
            processed_targets, _, _ = self._noise_injector(augmented_imgs, augmented_specs, corruption_type=corruption_types, corruption_params=corruption_params, noise_injection_factor=self.target_noise_injection_factor)
        else:
            processed_targets = augmented_imgs
        
        return processed_images, processed_targets, processed_spectrals, corruption_mask, imgs

    # Train Phase
    def _train_phase(self, train_loss=0.0, train_n=0.0, **kwargs):
        '''
        This function executes the training phase of the training epoch.

        Args:
            train_loss (float): Average training loss from the previous epoch.
            train_n (float): Number of image samples in the previous batch.
            inject_noise (bool): Boolean for corrupting images.
        
        Returns:
            train_loss (float): Average training loss from the current epoch.
            train_n (float): Number of image samples in the batch.
        '''
        # Do Data Minibatching 
        for batch in tqdm(self.train_loader, desc="Training Batch", total=len(self.train_loader), position=0, leave=True):
            imgs, specs = batch
            imgs, specs = imgs.to(self.device), specs.to(self.device)
            
            inps, targs, _, spec_mask, _= self._preprocessbatch(imgs, specs,
                                                                noisy_targets=kwargs.get("n2n", True),  
                                                                **kwargs)
            
            outputs = self.model(inps)

            if self.post_op:
                outputs = self._post_op(self.post_op, outputs, specs, spec_mask)
            
            loss = self.loss_fn(outputs, targs)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item() * inps.size(0)
            train_n += inps.size(0)
        
        train_loss = (train_loss / train_n) if train_n > 0 else 0.0
        return train_loss, train_n

    # Validate Phase
    def _valid_phase(self, valid_loss=0.0, valid_n=0.0, **kwargs):
        '''
        This function executes the Validation phase of the training epoch.

        Args:
            valid_loss (float): Average training loss from the previous epoch.
            valid_n (float): Number of image samples in the previous batch.
            inject_noise (bool): Boolean for corrupting images.
        
        Returns:
            valid_loss (float): Average training loss from the current epoch.
            valid_n (float): Number of image samples in the batch.
            average_psnr (float): Average PSNR score of the batch.
        '''
        # Validation Process
        self.model.eval()
        
        with torch.inference_mode():
            for idx, batch in tqdm(enumerate(self.valid_loader), desc="Validation Batch", total=len(self.valid_loader), leave=True):
                # inps, targs, _, _, origs = batch
                # inps, targs, origs = inps.to(self.device), targs.to(self.device), origs.to(self.device)
                imgs, specs = batch
                imgs, specs = imgs.to(self.device), specs.to(self.device)
                inps, targs, _, _, origs = self._preprocessbatch(imgs, specs, 
                                                                 noisy_targets=False, 
                                                                 **kwargs)
                
                outputs = self.model(inps)

                # if self.post_op == 'fspec':
                #     outputs = self._post_op(outputs, specs, spec_mask)
                
                loss = self.loss_fn(outputs, targs)
    
                valid_loss += loss.item() * inps.size(0)
                valid_n += inps.size(0)

                # Calculate PSNR scores
                avg_psnr = torch.mean(self._psnr_scores(outputs, targs))
                
                if idx == 0:
                    # 4-in-1 image + spectrum
                    assert inps.shape[0] >= 7, "Batch size too small. Minimum permitted size = 7"
                    prim = [x.squeeze(0) for x in [origs[6], inps[6], outputs[6], targs[6]]]
                    spec = [v for _, v in (self._compute_spectral(x, magnitude=True, normalize=False) for x in prim)]
                    pimg = torch.cat(prim, dim=1).add(0.5)
                    simg = torch.cat(spec, dim=1).mul(0.05)
                    img = torch.cat([pimg, simg], dim=0)

                    # Saving the Outputs.
                    save_image(img, self.training_test_path / f"img{self.epoch:03d}.png", normalize=False)
                    
        valid_loss = (valid_loss / valid_n) if valid_n > 0 else 0.0
        return valid_loss, valid_n, avg_psnr.item()

    # Last Epoch Phase
    def _final_epoch(self, **kwargs):
        '''
        This function executes the Final Epoch phase of the training epoch.
        '''
        print("Final Epoch: \n")
        # Do a full Validation set testing with result saving.
        loader = DataLoader(Subset(self.valid_data, indices=range(10)), batch_size=1, shuffle=False, pin_memory=True, pin_memory_device="cuda")

        self.model.eval()
        with torch.inference_mode():
            psnr_file = self.validation_test_path / "PSNR.txt"
            with psnr_file.open('wt') as fout:
                fout.write(f'Sr.no.:\tOriginal\tInput\t\tOutput\t\tGains\n' +
                           '-'*70 + '\n')
                for idx, batch in tqdm(enumerate(loader), desc="Validation Testing", total=len(loader), leave=True):
                    imgs, specs = batch
                    imgs, specs = imgs.to(self.device), specs.to(self.device)
                    inp, targ, _, _, orig = self._preprocessbatch(imgs, specs,
                                                                 noisy_targets=False,
                                                                 **kwargs)
                    denoised_op = self.model(inp)
                    
                    # 4-in-1 image + spectrum
                    prim = [x.squeeze(0) for x in [orig, inp, denoised_op, targ]]
                    spec = [v for _, v in (self._compute_spectral(x, magnitude=True, normalize=False) for x in prim)]
                    pimg = torch.cat(prim, dim=1).add(0.5)
                    simg = torch.cat(spec, dim=1).mul(0.05)
                    img = torch.cat([pimg, simg], dim=0)

                    # Saving the Outputs.
                    save_image(img, self.validation_test_path / f"final{idx:03d}.png", normalize=False)
                    # save_image((torch.clip(orig, -0.5, 0.5) + 0.5).cpu(), self.validation_test_path / f"original{idx:03d}.png")
                    save_image((torch.clip(denoised_op, -0.5, 0.5).add(0.5)).cpu(), self.validation_test_path / f"output{idx:03d}.png")
                    save_image((torch.clip(targ, -0.5, 0.5).add(0.5)).cpu(), self.validation_test_path / f"target{idx:03d}.png")
                    save_image((torch.clip(inp, -0.5, 0.5).add(0.5)).cpu(), self.validation_test_path / f"input{idx:03d}.png")
                    
                    # PSNR Scores
                    orig_vs_targ = self._psnr_scores(imgs, targ).item()
                    op_vs_targ = self._psnr_scores(denoised_op, targ).item()
                    inp_vs_targ = self._psnr_scores(inp, targ).item()
                    
                    fout.write(f'{idx:02d}:\t{orig_vs_targ:0.5f}\t{inp_vs_targ:0.5f}\t{op_vs_targ:0.5f}\t{(op_vs_targ - orig_vs_targ):0.5f}\n')

    # Generate Example
    def _generate_example(self, **kwargs):
        '''
        This function generates an example sample pair of the noisy input and target images for the active training experiment.
        It also saves a labeled and an unlabeled versions of the sample for the report.
        
        Args:
            inject_noise (bool) : Boolean to inject noise onto images.
            **kwargs: Keyword arguments for respective noise types.
        
        Returns:
            corrupted_image (Tensor): Sample of the corrupted image.
        '''
        # Save image sample
        imgs, specs = next(iter(self.train_loader))
        assert imgs.shape[0] >= 10, "Batch size too small. Minimum permitted size = 10"
        imgs, specs = torch.narrow(imgs, 0, 9, 1), torch.narrow(specs, 0, 9, 1)
        imgs, specs = imgs.to(self.device), specs.to(self.device)
        dummy_ip, t, s, s_m, o= self._preprocessbatch(imgs, specs, 
                                                     noisy_targets=kwargs.get("n2n", True), 
                                                     **kwargs)

        prim = [x.squeeze(0) for x in [dummy_ip, t]]
        spec = [v for _, v in (self._compute_spectral(x, magnitude=True, normalize=False) for x in prim)]
        pimg = torch.cat(prim, dim=1).add(0.5)
        simg = torch.cat(spec, dim=1).mul(0.05)
        img = torch.cat([pimg, simg], dim=0)
        save_image(img.cpu(), self.results_dir / "2x2_example.png", normalize=False)

        imgs = [np.clip(x.squeeze(0).add(0.5).cpu().numpy().astype(np.float32), 0.0, 1.0) for x in prim]
        specs = [np.clip(x.squeeze(0).mul(0.05).cpu().numpy().astype(np.float32), 0.0, 1.0) for x in spec]
        fig, axes = plt.subplots(2, 2)

        # Titles for columns
        axes[0, 0].set_title("Noisy", fontsize=12)
        axes[0, 1].set_title("Target", fontsize=12)
        
        # Plot images
        axes[0, 0].imshow(imgs[0], cmap='gray')
        axes[0, 1].imshow(imgs[1], cmap='gray')
        axes[1, 0].imshow(specs[0], cmap='gray')
        axes[1, 1].imshow(specs[1], cmap='gray')
        
        # Clean up axes
        for ax in axes.ravel():
            ax.axis('off')

        # Add row labels using fig.text
        fig.text(0.08, 0.75, "Spatial", va='center', ha='left', fontsize=12, rotation='vertical')
        fig.text(0.08, 0.28, "Spectral", va='center', ha='left', fontsize=12, rotation='vertical')
        plt.tight_layout()
        plt.savefig(self.results_dir / "2x2_example_labled.png", dpi=300, bbox_inches='tight')
        plt.show()
        return dummy_ip
        
    # Main
    def train(self, config):
        # will contain the actual train loop.
        '''
        This function implements the training loop.
        
        Args:
            **kwargs:
                train_config (Dict): Contains all the parameters for the current run.
        '''

        # Save config
        self._save_config(**config)
        
        # Start Logging
        self.logger.start()
        
        print("Training Started...")
        print(f"Total Epochs: {self.max_epochs} \nBatch Size: {self.train_loader.batch_size} \nInitial Learning Rate: {optimizer.param_groups[0]['lr']}\n")

        # Generate a sample of training input-target pairs.
        dummy_ip = self._generate_example(**config)
        
        # Add Model graph to the Tensorboard.
        self.writer.add_graph(model=self.model, input_to_model=dummy_ip.to(self.device), verbose=False)

        # Clock Training start time.
        train_start_time = time.time()

        # Training Loop
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            # Clocking Epoch start time
            start_time = time.time()
            
            # Training Phase
            train_loss, train_n = self._train_phase(**config)
            
            # Validation Phase
            valid_loss, valid_n, avg_psnr = self._valid_phase(**config)

            # Update LR Scheduler step
            self.scheduler.step()

            # Calculating time elapsed for the current epoch.
            epoch_time = time.time() - start_time
            # Update Tensorboard Summary
            self.writer.add_scalar("Training/Loss", train_loss, global_step=epoch, new_style=True)
            self.writer.add_scalar("Validation/Loss", valid_loss, global_step=epoch, new_style=True)
            self.writer.add_scalar("Validation/Average_PSNR", avg_psnr, global_step=epoch, new_style=True)
            self.writer.add_scalar("Learning_rate", self.optimizer.param_groups[0]["lr"], global_step=epoch, new_style=True)
            self.writer.add_scalar("Training/time-per-epoch", epoch_time, global_step=epoch, new_style=True)
            print(f'[{self.device}]Epoch [{epoch+1}/{self.max_epochs}] | Time: {epoch_time: 0.2f} | Train Loss: {train_loss: 0.6f} | Validation Loss: {valid_loss: 0.6f} | Avg. PSNR: {avg_psnr: 0.6f} | Learning Rate: {self.optimizer.param_groups[0]["lr"]: 0.10f}')
            
             # Final Epoch
            if epoch == self.max_epochs-1:
                # Last Epoch Phase
                self.final_epoch = True
                self._final_epoch(**config)

            # Save Snapshot
            if (epoch%self.save_interval == self.save_interval-1 or self.final_epoch) and self.device == 0:
                self._save_snapshot()
        
        # Close the Tensorboard Summary Writer
        self.writer.close()
        
        # Calculate the Elapsed Time for the Training Loop
        total_seconds = time.time() - train_start_time
        print(f"Time Elapsed: {int(total_seconds // 3600)}hrs : {int((total_seconds % 3600) // 60)}mins : {int(total_seconds % 60)}secs." )
        
        # Stop with the logging
        self.logger.stop()

def main(setup_config, **config):
    # setup_config - Contains run related params like dir, paths, 
    # config - Contains model training related params like max_epocs, corruption_type, etc
    max_epochs = config.get('max_epochs', 300)
    model = Noise2Noise()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)
    scheduler = lr_sch.SequentialLR(optimizer=optimizer, 
                                schedulers=[lr_sch.LinearLR(optimizer=optimizer, start_factor=0.001, total_iters=int(max_epochs * 0.1)),
                                            lr_sch.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=int(max_epochs * 0.6)), 
                                            lr_sch.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.001, total_iters=int(max_epochs * 0.3))], 
                                milestones=[int(max_epochs * 0.1), int(max_epochs * 0.7)])
    load_datasets(max_epochs=max_epochs, ds_dir=setup_config.get("ds_dir"))

def load_datasets(max_epochs:int, ds_dir:Path):
    train_dataset = torch.load(ds_dir / "HQ_train_dataset_20.pth", weights_only=False)
    valid_dataset = torch.load(ds_dir / "HQ_valid_dataset_5.pth", weights_only=False)
    # eval_dataset = torch.load(ds_dir / "HQ_eval_dataset_10.pth", weights_only=False)
    return train_dataset, valid_dataset
