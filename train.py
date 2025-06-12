import os
import gc
import json
import csv
import time
import random
import warnings
import numpy as np
from easydict import EasyDict
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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dataset import CTScans
from network import Noise2Noise
from utils import NoiseType, get_unique_path, Logger

class Trainer():
    def __init__(self,
                 device: int,
                 model: nn.Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 save_interval:int = 10,
                 setup_config:EasyDict=None,
                 train_config:Optional[EasyDict]=None
                 ) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        self.save_interval = save_interval
        self.config = train_config
        self.setup_config = setup_config
        self.post_op = setup_config.post_op
        self.isfinal = False
        self.start_epoch = 0
        self.epoch = 0
        self.results_dir = setup_config.results_dir
        self.training_test_path = self.results_dir / "Training Tests"
        self.validation_test_path = self.results_dir / "Post Training Tests"
        self.snapshot_path = self.results_dir / "Snapshots"
        self.snapshot_name = 'latest-snapshot'
        if self.device == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.training_test_path.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.mkdir(parents=True, exist_ok=True)
            self.validation_test_path.mkdir(parents=True, exist_ok=True)
            if any(self.validation_test_path.iterdir()):
                self.validation_test_path = get_unique_path(base_path = self.validation_test_path)
            self.validation_test_path.mkdir(parents=True, exist_ok=True)
            
        self.writer = SummaryWriter(log_dir=str(self.results_dir))
        self.logger = Logger(str(self.results_dir / "log.txt"))
        self.augment_translate_cache = {}
    
    # Save config information
    def _save_config(self):
        '''
        This function saves the parameters and hyperparameters used for the current run of the experiment to 'run-config.txt'.
        '''
        logger = Logger(str(self.results_dir / "run-config.txt"), init_log=True)
        config = {
            'loss': str(self.loss_fn),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'post_op': self.post_op,
            'max_epochs': self.setup_config.max_epochs,
            'start_epoch': self.start_epoch,
            'results_dir': str(self.results_dir),
            'corrupt_targets': self.config.n2n,
            'batch_size': self.setup_config.batch_size,
            'checkpoint_save_interval': self.save_interval,
            'augment_params': self.config.augment_params,
            'corruption_types': [noise.value for noise in self.config.corruption_types],
            'corruption_params': self.config.corruption_params
        }
        
        logger.start()
        logger.log("############ Config #############")
        logger.log(json.dumps({'config': config}, indent=4))
        logger.log("#################################\n")
        logger.stop()
    
    # Save model snapshot
    def _save_snapshot(self, epoch:int) -> None:
        snapshot = {}
        snapshot["STATE_DICT"] = self.model.state_dict()
        snapshot["CURR_EPOCH"] = epoch
        snapshot["OPTIMIZER_DICT"] = self.optimizer.state_dict()

        torch.save(snapshot, self.snapshot_path / 'latest-snapshot.pth')
        torch.save(snapshot, self.snapshot_path / f'snapshot-{epoch+1}.pth')
        self.logger.log(f"Epoch {epoch}:\n\t Training Snapshot Saved at as [snapshot-{epoch+1}.pth] at {self.snapshot_path} \n\t Latest Snapshot Updated!")

    # Load model snapshot
    def load_snapshot(self, snapshot_path:Path=None, snapshot_name:str=''):
        if snapshot_name == '':
            snapshot_name = self.snapshot_name
        if snapshot_path is None:
            snapshot_path = self.snapshot_path
            
        path = snapshot_path / (snapshot_name + ".pth")
        assert path.exists()
        
        snapshot = torch.load(path, weights_only=True)
        epoch = snapshot["CURR_EPOCH"]
        self.model.load_state_dict(snapshot["STATE_DICT"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_DICT"])
        self.logger.log(f"Loading Snapshot: %s.pth | Current Epoch: %d"%(self.snapshot_name, epoch+1))
        self.start_epoch = epoch+1
    
    # Generate Example
    def _generate_example(self):
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
        assert imgs.shape[0] >= 1, "Batch size too small. Minimum permitted size = 1"
        if imgs.shape[0] >= 10:
            imgs, specs = torch.narrow(imgs, 0, 9, 1), torch.narrow(specs, 0, 9, 1)
        else:
            imgs, specs = torch.narrow(imgs, 0, 0, 1), torch.narrow(specs, 0, 0, 1)
        imgs, specs = imgs.to(self.device), specs.to(self.device)
        dummy_ip, t, s, s_m, o= self._preprocessbatch(imgs, specs, noisy_targets=self.config.get("n2n", True))

        prim = [x.squeeze(0) for x in [dummy_ip, t]]
        spec = [v for _, v in (self._compute_spectral(x, shift=True, magnitude=True, scaled=True, normalize=False) for x in prim)]
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
        # plt.show()
        plt.close()
        return dummy_ip

    # PSNR Scores
    def _psnr_scores(self, image, target, max_pixel_value=1.0):
        '''
        Computed PSNR scores for given image and target pair.

        Args:
            image (Tensor): Image to calculate the score for.
            target (Tensor): Image as a reference target.
            max_pixel_value (float): Max value of 
            
        Returns:
            spectral_image (Tensor): The spectral representation.
        '''
        # max_pixel_value = 1.0 as the input is in range [-0.5, 0.5], making the amplitude/range of values to be 1.0 
        assert len(image.shape) == 3 and len(target.shape) == 3
        image = torch.clip(image, -0.5, 0.5).add(0.5)
        target = torch.clip(target, -0.5, 0.5).add(0.5)
        
        mse = torch.clip(torch.mean((target - image)**2, dim=(-2, -1)), min=1e-8)
        psnr = 10.0 * torch.log10(max_pixel_value**2 / mse)
        return psnr
    
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
    def _compute_spectral(self, imgs:torch.Tensor, shift:bool=False, ifftshift:bool=False, magnitude:bool=False, scaled:bool=False, normalize:bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    def _compute_spatial(self, specs:torch.Tensor, shift:bool=True, **kwargs) -> torch.Tensor:
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
                shifted_specs = torch.fft.ifftshift(specs, dim=(-2, -1))
            else:
                shifted_specs = torch.fft.fftshift(specs, dim=(-2, -1))
            output_imgs = torch.fft.ifft2(shifted_specs).real.to(torch.float32)
        else:
            output_imgs = torch.fft.ifft2(specs).real.to(torch.float32)
        
        return output_imgs
    
    # Inject Noise
    corruption_masks={}
    def _inject_noise(self, imgs:torch.Tensor, specs:torch.Tensor, corruption_type:NoiseType, noise_injection_factor:float):
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
        corruption_params = self.config.corruption_params
        noise_injection_factor = corruption_params.get("noise_injection_factor", 1.0)
        persistant_noise = corruption_params.get("persistant_noise", False)

        output_imgs, output_specs, output_mask = torch.zeros_like(imgs, dtype=torch.float32), torch.zeros_like(specs, dtype=torch.complex64), torch.ones_like(imgs)
        # Choose to corrupt or not, and choose the corruption type.
        if corruption_type == NoiseType.NO_NOISE:
            output_imgs, output_specs, output_mask = imgs, specs, torch.zeros_like(imgs)

        elif corruption_type == NoiseType.GAUSSIAN:
            gaussian_params = corruption_params.get("gaussian_params", dict())
            if "std" not in gaussian_params:
                warnings.warn("Missing Parameter Warning: Using default values for - gaussian_params:{\"Gaussian Standard Deviation\" = 25}")
            std = gaussian_params.get("std", 25) * noise_injection_factor
            std /= 255.0
            
            if persistant_noise:
                # cache_key = (self.epoch%unique_key_freq, "gaussian", std)
                # if cache_key not in self.corruption_masks:
                #     mask = torch.normal(mean=0.0, std=std, size=imgs[0].shape, device=self.device)
                
                # Use mask form cache if it exists, else sample a new mask and store it in cache.
                mask = torch.normal(mean=0.0, std=std, size=imgs[0].shape, device=self.device)  # Needs to be changed !!
            else:
                mask = torch.normal(mean=0.0, std=std, size=imgs[0].shape, device=self.device)
            output_imgs = imgs + mask
            output_specs = self._compute_spectral(output_imgs, shift=True)
            
            mask = torch.log1p(self._compute_spectral(mask, shift=True).abs())
            output_mask = (mask - mask.min())/(mask.max() - mask.min())
            output_mask *= -1
            
        elif corruption_type == NoiseType.POISSON:
            poisson_params = corruption_params.get("poisson_params", dict())
            if "strength" not in poisson_params:
                warnings.warn("Missing Parameter Warning: Using default values for - poisson_params:{\"poisson_strength\" = 30}")
            lam_max = poisson_params.get("strength", 30) * (2.0 - noise_injection_factor)
            
            if "distribution" not in poisson_params:
                warnings.warn("Missing Parameter Warning: Using default values for - poisson_params:{\"distribution\" = \"uniform\"}")
            distribution = poisson_params.get("distribution", "uniform")
            
            mask = None
            criterion = None
            lam_min = 10
            if persistant_noise:
                # cache_key = (self.epoch%unique_key_freq, "poisson", corruption_strength, distribution)
                # if cache_key not in self.corruption_masks:
                #     # Create a corruption mask
                #     if distribution == "uniform":
                #         if "mask_ratio" not in kwargs["corruption_params"]:
                #             warnings.warn("Missing Parameter Warning: Using default values for - corruption_params:{\"mask_ratio\" = 0.05}")
                #         criterion = kwargs["corruption_params"].get("mask_ratio", 0.05)
                        
                #     elif distribution == "gaussian":
                #         if "sigma" not in kwargs["corruption_params"]:
                #             raise ValueError("Missing 'sigma' in corruption_params.")
                            
                #         sigma = kwargs["corruption_params"].get("poisson_sigma", 1.0)
                #         y = torch.arange(H, dtype=torch.float32, device=device) - H // 2
                #         x = torch.arange(W, dtype=torch.float32, device=device) - W // 2
                #         yy, xx = torch.meshgrid(y, x, indexing="ij")
                #         criterion = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                        
                #     else:
                #         # Default to uniform distribution with mask_ratio = 0.05
                #         criterion = 0.05
                        
                #     mask = (torch.rand((H, W), device=self.device) < criterion).float()
                #     self.corruption_masks[cache_key] = mask
                # else:
                #     mask = self.corruption_masks[cache_key]
                # corruption_strength = corruption_mask[cache_key]
                
                # If previous corruption_strength value exists in cache, use it else Sample a random value and store it in cache.
                corruption_strength = torch.empty(1).uniform_(lam_min, lam_max).item() # Needs to be changed!
            else:
                # Pick a random corruption_strength value
                corruption_strength = torch.empty(1).uniform_(lam_min, lam_max).item()
            
            if distribution == "uniform":
                if "mask_ratio" not in poisson_params:
                    warnings.warn("Missing Parameter Warning: Using the default values for - poisson_params:{\"mask_ratio\" = 1.0}")
                criterion = poisson_params.get("mask_ratio", 1.0)
            elif distribution == "gaussian":
                if "sigma" not in poisson_params:
                    raise ValueError("Missing 'sigma' in corruption_params.")
                sigma = poisson_params.get("poisson_sigma", 1.0)
                
                y = torch.arange(H, dtype=torch.float32, device=device) - H // 2
                x = torch.arange(W, dtype=torch.float32, device=device) - W // 2
                yy, xx = torch.meshgrid(y, x, indexing="ij")
                criterion = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            else:
                raise ValueError("Incorrect noise distribution pattern provided.")
            
            poisson_noise = torch.poisson(torch.clip(torch.abs(imgs + 0.5), 0.0, 1.0) * corruption_strength) / corruption_strength - 0.5
            mask = (torch.rand((H, W), device=self.device) < criterion).float()
            batch_mask = mask.unsqueeze(0)
            output_imgs = torch.where(batch_mask.bool(), poisson_noise, imgs)
            output_specs = self._compute_spectral(output_imgs, shift=True)
            
            mask = torch.log1p(self._compute_spectral(mask, shift=True).abs())
            output_mask = (mask - mask.min())/(mask.max() - mask.min())  # Normalize the mask.
            output_mask *= -1  # Invert the mask for post-op
        
        elif corruption_type == NoiseType.BERNOULLI:
            bspec_params = corruption_params.get("bspec_params", dict())
            if "p_edge" not in bspec_params:
                    warnings.warn("Missing Parameter Warning: Using default values for - bernoulli_params:{\"p_edge\" = 0.05}")
            p_edge = np.clip(bspec_params.get("p_edge", 0.05) * (2.0 - noise_injection_factor), 0.0, 1.0)
            
            # cache_key = (self.epoch%unique_key_freq, "bernoulli", p_edge)
            # if cache_key not in self.corruption_masks:
            #     # Create a corruption mask
            #     y = torch.arange(H, dtype=torch.float32, device=self.device) - H // 2
            #     x = torch.arange(W, dtype=torch.float32, device=self.device) - W // 2
            #     yy, xx = torch.meshgrid(y**2, x**2, indexing='ij')
            #     r_dist = torch.sqrt(xx + yy)
            #     prob_mask = (p_edge ** (2.0 / W)) ** r_dist
                
            #     keep = (torch.rand(size=(H, W), device=self.device, dtype=torch.float32) ** 2) < prob_mask
            #     keep = keep & torch.flip(keep, dims=[0, 1])
                
            #     self.corruption_masks[cache_key] = (keep, prob_mask)
                
            # else:    
            #     keep, prob_mask = self.corruption_masks[cache_key]

            keep = None
            prob_mask = None
            if persistant_noise:
                # If mask exists in cache, use it else sample a new mask and cache it.
                pass
            else:
                # Create a corruption mask
                y = torch.arange(H, dtype=torch.float32, device=self.device) - H // 2
                x = torch.arange(W, dtype=torch.float32, device=self.device) - W // 2
                yy, xx = torch.meshgrid(y**2, x**2, indexing='ij')
                r_dist = torch.sqrt(xx + yy)
                prob_mask = (p_edge ** (2.0 / W)) ** r_dist
                
                keep = (torch.rand(size=(H, W), device=self.device, dtype=torch.float32) ** 2) < prob_mask
                keep = keep & torch.flip(keep, dims=[0, 1])
            
            # Apply Mask
            mskd_specs = specs * keep
            output_mask = keep.to(torch.float32)
            output_specs = torch.fft.ifftshift(mskd_specs / torch.where(keep, prob_mask, 1e-8), dim=(-2, -1))
            output_imgs = torch.fft.ifft2(output_specs).real.float()
            
            output_mask *= -1

        elif corruption_type == NoiseType.GAUSSIAN_BLUR:
            blur_params = corruption_params.get("blur_params", dict())
            if "kernel" not in blur_params:
                warnings.warn("Missing Parameter Warning: Using default values for - blur_params:{\"kernel\" = 15}")
            kernel = blur_params.get("kernel", 15)
            
            if "sigma" not in blur_params:
                warnings.warn("Missing Parameter Warning: Using default values for - blur_params:{\"sigma\" = 1.0}")
            sigma = blur_params.get("sigma", 1.0) * noise_injection_factor

            sigma = random.uniform(0.5, sigma)
            rec_kernel = int(2*np.ceil(sigma*3)) + 1
            kernel_size = random.choice([k for k in range(rec_kernel, min(kernel, 2*rec_kernel + 1), 2)])
            
            output_imgs = gaussian_blur(img=imgs, kernel_size=kernel_size, sigma=sigma)
            output_specs = self._compute_spectral(output_imgs, shift=True)
            output_mask = torch.zeros_like(output_imgs)
        
        else:
            raise ValueError(f"Requested an invalid corruption type - [\"{corruption_type.name}\"]")
        
        return output_imgs, output_specs, output_mask
    
    # Noise Injector
    def _noise_injector(self, imgs:torch.Tensor, specs:torch.Tensor, corruption_types:List[NoiseType], noise_injection_factor=1.0):
        '''
            This function sequentially injects noise to a tensor of images.

            Args:
                imgs (Tensor): Images to add noise to.
                specs (Tensor): Spectral representation of the images.
                corruption_types (List[str]): List of corruptions to add in sequence.
            
            Returns:
                corrupted_images (Tensor): Corrupted images.
                corrupted_spectrals (Tensor): The spectral representation.
                corruption_mask (Tensor): The effective corruption mask for the sequence of noise patterns.
        '''
        output_imgs, output_specs, output_masks = imgs, specs, torch.zeros_like(imgs)
        if corruption_types:
            for corruption in corruption_types:
                output_imgs, output_specs, masks = self._inject_noise(output_imgs, output_specs, corruption_type=corruption, noise_injection_factor=noise_injection_factor)
                if corruption == NoiseType.BERNOULLI:
                    output_masks += torch.ones_like(imgs) + masks
            
        return output_imgs, output_specs, output_masks
     
    # Preprocess Batch
    def _preprocessbatch(self, imgs:torch.Tensor, specs:torch.Tensor, noisy_targets:bool=True):
        '''
        This function performs preprocessing on a batch of images.

        Args:
            imgs (Tensor): Batch of images to process.
            specs (Tensor): Spectral representation of images.
            noisy_targets (bool): Boolean to select noisy or clean targets.
        
        Returns:
            processed_images (Tensor): Images after processing.
            processed_targets (Tensor): Targets for training.
            processed_spectrals (Tensor): The spectral representation.
            corruption_mask (Tensor: float): The effective mask used for injecting noise values.
            original_images (Tensor): Original images before noise injection. (Post-Augmentation Images)
        '''
        inject_noise = self.config.get("inject_noise", True)
        corruption_types = self.config.get("corruption_types", [NoiseType.NO_NOISE])
        corruption_params = self.config.get("corruption_params")
            
        # Augment Images
        augmented_imgs, augmented_specs = self._augment_data(imgs, specs, augment_params=self.config.get("augment_params", dict()))

        # Inject Noise
        if inject_noise:
            processed_images, processed_spectrals, corruption_mask = self._noise_injector(augmented_imgs, augmented_specs, corruption_types=corruption_types)
        else:
            processed_images, processed_spectrals, corruption_mask = augmented_imgs, augmented_specs, torch.ones_like(augmented_imgs)

        # Corrupt Targets
        if noisy_targets:
            processed_targets, _, _ = self._noise_injector(augmented_imgs, augmented_specs, corruption_types=corruption_types, noise_injection_factor=self.config.corruption_params.target_noise_injection_factor)
        else:
            processed_targets = augmented_imgs
        
        return processed_images, processed_targets, processed_spectrals, corruption_mask, augmented_imgs
    
    # Post-op
    def _post_op(self, denoised, spec_value, spec_mask):
        '''
        Performs the post-operation procedure of forcing known frequencies before the training.

        Args:
            denoised (Tensor: float): The image to perform post-op on.
            spec_value (Tensor: complex): The spectral representation of the image.
            spec_mask (Tensor: float): The mask used for post-op in spectral domain.
            
        Returns:
            denoised (Tensor: float): The image after post-op is performed.
        '''
        op_type = self.setup_config.get('post_op', None)
        
        if op_type =='fspec':
            # print("Force denoised spectrum to known values.")
            # FFT
            # denoised_spec = torch.fft.fft2(denoised)     
            # FFT shift
            denoised_spec = self._compute_spectral(denoised, shift=True) #torch.fft.fftshift(denoised_spec)
            # Ensure correct dtypes and device
            spec_value = spec_value.to(denoised_spec.dtype).to(denoised_spec.device)
            spec_mask = spec_mask.to(denoised_spec.dtype).to(denoised_spec.device)
            # Force known frequencies using mask
            denoised_spec = spec_value * spec_mask + denoised_spec * (1. - spec_mask)
            # Shift back and IFFT
            # denoised = torch.fft.ifft2(torch.fft.ifftshift(denoised_spec)).real
            denoised = self._compute_spatial(denoised_spec, shift=True)
        
        elif op_type =='fspec_old':
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
            denoised_spec = torch.fft.fft2(denoised)     
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
            warnings.warn("Invalid Post-Op requested. No Post-Op performed.")
        return denoised
    
    # Training Phase
    def _train_phase(self, train_loss=0.0, train_n=0.0):
        '''
        This function executes the training phase of the training epoch.

        Args:
            train_loss (float): Average training loss from the previous epoch.
            train_n (float): Number of image samples in the previous batch.
        
        Returns:
            train_loss (float): Average training loss from the current epoch.
            train_n (float): Number of image samples in the batch.
        '''
        # Do Data Minibatching 
        for batch in tqdm(self.train_loader, desc="Training Batch", total=len(self.train_loader), position=0, leave=True):
            imgs, specs = batch
            imgs, specs = imgs.to(self.device), specs.to(self.device)
            
            inps, targs, _, spec_mask, _= self._preprocessbatch(imgs, specs, noisy_targets=self.config.get("n2n", True))
            
            outputs = self.model(inps)

            if self.post_op:
                outputs = self._post_op(outputs, specs, spec_mask)
            
            loss = self.loss_fn(outputs, targs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item() * inps.size(0)
            train_n += inps.size(0)
        
        train_loss = (train_loss / train_n) if train_n > 0 else 0.0
        return train_loss, train_n
    
    # Validate Phase
    def _valid_phase(self, valid_loss=0.0, valid_n=0.0):
        '''
        This function executes the Validation phase of the training epoch.

        Args:
            valid_loss (float): Average training loss from the previous epoch.
            valid_n (float): Number of image samples in the previous batch.
        
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
                inps, targs, _, _, origs = self._preprocessbatch(imgs, specs, noisy_targets=False)
                
                outputs = self.model(inps)
                
                loss = self.loss_fn(outputs, targs)
    
                valid_loss += loss.item() * inps.size(0)
                valid_n += inps.size(0)

                # Calculate PSNR scores
                avg_psnr = torch.mean(self._psnr_scores(outputs, targs))
                
                if idx == 0 and self.device == 0:
                    # 4-in-1 image + spectrum
                    assert inps.shape[0] >= 1, "Batch size too small. Minimum permitted size = 1"
                    if inps.shape[0] >= 7:
                        prim = [x.squeeze(0) for x in [origs[6], inps[6], outputs[6], targs[6]]]
                    else:
                        prim = [x.squeeze(0) for x in [origs[0], inps[0], outputs[0], targs[0]]]
                    
                    spec = [v for _, v in (self._compute_spectral(x, shift=True, magnitude=True, scaled=True, normalize=False) for x in prim)]
                    pimg = torch.cat(prim, dim=1).add(0.5)
                    simg = torch.cat(spec, dim=1).mul(0.05)
                    img = torch.cat([pimg, simg], dim=0)

                    # Saving the Outputs.
                    save_image(img, self.training_test_path / f"img{self.epoch:03d}.png", normalize=False)
                    
        valid_loss = (valid_loss / valid_n) if valid_n > 0 else 0.0
        return valid_loss, valid_n, avg_psnr.item()
    
    # Last Epoch Phase
    def _final_epoch(self):
        '''
        This function executes the Final Epoch phase of the training epoch.
        '''
        self.logger.log("Final Epoch: \n")
        # Do a full Validation set testing with result saving.
        loader = DataLoader(Subset(self.valid_loader.dataset, indices=range(min(10, len(self.valid_loader.dataset)))), batch_size=1, shuffle=False, pin_memory=True)

        self.model.eval()
        with torch.inference_mode():
            psnr_file = self.validation_test_path / "PSNR.csv"
            with psnr_file.open('w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Original_vs_Target', 'Input_vs_Target', 'Output_vs_Target', 'Gains'])
                for idx, batch in tqdm(enumerate(loader), desc="Validation Testing", total=len(loader), leave=True):
                    imgs, specs = batch
                    imgs, specs = imgs.to(self.device), specs.to(self.device)
                    inp, targ, _, _, orig = self._preprocessbatch(imgs, specs, noisy_targets=False)
                    denoised_op = self.model(inp)
                    
                    # 4-in-1 image + spectrum
                    prim = [x.squeeze(0) for x in [orig, inp, denoised_op, targ]]
                    spec = [v for _, v in (self._compute_spectral(x, shift=True, magnitude=True, scaled=True, normalize=False) for x in prim)]
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

                    writer.writerow([f'{idx:02d}', 
                                     f'{orig_vs_targ:0.5f}', 
                                     f'{inp_vs_targ:0.5f}', 
                                     f'{op_vs_targ:0.5f}', 
                                     f'{(op_vs_targ - orig_vs_targ):0.5f}'])
    
    # Run one Epoch
    def _run_epoch(self, epoch:int):
        # Clocking Epoch start time
        start_time = time.time()

        # Training
        train_loss, train_n = self._train_phase()

        # Validation
        valid_loss, valid_n, avg_psnr = self._valid_phase()

        # Take the LR Scheduler Step
        self.scheduler.step()
        
        # Calculating time elapsed for the current epoch.
        epoch_time = time.time() - start_time
        
        if self.device == 0:
            # Update Tensorboard Summary
            self.writer.add_scalar("Training/Loss", train_loss, global_step=epoch, new_style=True)
            self.writer.add_scalar("Validation/Loss", valid_loss, global_step=epoch, new_style=True)
            self.writer.add_scalar("Validation/Average_PSNR", avg_psnr, global_step=epoch, new_style=True)
            self.writer.add_scalar("Learning_rate", self.optimizer.param_groups[0]["lr"], global_step=epoch, new_style=True)
            self.writer.add_scalar("Training/time-per-epoch", epoch_time, global_step=epoch, new_style=True)
            
        # Log status
        self.logger.log(f'[{self.device}]Epoch [{epoch+1}/{self.setup_config.max_epochs}] | Time: {epoch_time: 0.2f} | Train Loss: {train_loss: 0.6f} | Validation Loss: {valid_loss: 0.6f} | Avg. PSNR: {avg_psnr: 0.6f} | Learning Rate: {self.optimizer.param_groups[0]["lr"]: 0.10f}')
    
    # Training Loop
    def _train(self, max_epochs:int):
        # Start Logging
        self.logger.start()
        
        if self.device == 0:
            self._save_config()
            if self.setup_config.load_checkpoint:
                self.logger.log(f"Loaded Snapshot - [{self.setup_config.snapshot_name}.pth]")
            self.logger.log("Training Started...")
            self.logger.log(f"Total Epochs: {max_epochs} \nBatch Size: {self.train_loader.batch_size} \nInitial Learning Rate: {self.optimizer.param_groups[0]['lr']}\n")
            # Generate a sample of training input-target pairs.
            dummy_ip = self._generate_example()
            # Add Model graph to the Tensorboard.
            self.writer.add_graph(model=self.model, input_to_model=dummy_ip.to(self.device), verbose=False)
            
        train_start_time = time.time()
        for epoch in range(self.start_epoch, max_epochs):
            self.epoch = epoch
            self._run_epoch(epoch)

            if epoch == self.setup_config.max_epochs-1:
                self.isfinal = True
                if self.device == 0:
                    self._final_epoch()
            # Save Snapshot
            if (epoch % self.save_interval == self.save_interval-1 or self.isfinal) and self.device == 0:
                self._save_snapshot(epoch)

        total_seconds = time.time() - train_start_time
        if self.device == 0:
            # Stop Tensorboard summary 
            self.writer.close()
            # Calculate Elapsed Time for the Training Loop
            self.logger.log(f"Time Elapsed: {int(total_seconds // 3600)}hrs : {int((total_seconds % 3600) // 60)}mins : {int(total_seconds % 60)}secs.\n\n" )
            
        # Stop with the logging
        self.logger.stop()
    

def load_dataset(ds_path: Path) -> CTScans:
    return torch.load(ds_path, weights_only=False)

def prepare_dataloader(dataset: CTScans, batch_size:int, shuffle:bool=False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle, pin_memory_device='cuda')

def main(setup_config:EasyDict, config:EasyDict):
    # setup_config - Contains run related params like dir, paths, max_epochs, lr, etc.
    # config - Contains model training related params like corruption_type, corruption_params, etc.
    model = Noise2Noise()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=setup_config.lr)
    loss_fn = nn.MSELoss()
    
    max_epochs = setup_config.max_epochs
    scheduler = None
    if setup_config.lr_scheduler_type == 'plateau':
        scheduler = lr_sch.SequentialLR(optimizer=optimizer, 
                                        schedulers=[lr_sch.LinearLR(optimizer=optimizer, start_factor=0.001, total_iters=int(max_epochs * 0.1)), 
                                                    lr_sch.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=int(max_epochs * 0.6)),  
                                                    lr_sch.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.001, total_iters=int(max_epochs * 0.3))], 
                                        milestones=[int(max_epochs * 0.1), int(max_epochs * 0.7)])
    
    # Load Datasets
    train_dataset = load_dataset(setup_config.train_dataset_path)
    valid_dataset = load_dataset(setup_config.valid_dataset_path)

    # Data Loaders
    train_loader = prepare_dataloader(dataset=train_dataset, batch_size=setup_config.batch_size, shuffle=True)
    valid_loader = prepare_dataloader(dataset=valid_dataset, batch_size=setup_config.batch_size, shuffle=False)
    
    # Initialize Trainer()
    trainer = Trainer(device=0,
                      model=model,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      scheduler=scheduler,
                      setup_config=setup_config,
                      train_config=config
                     )
    if setup_config.load_checkpoint:
        snapshot_path = setup_config.get('snapshot_path', None)
        snapshot_name = setup_config.get('snapshot_name', '')
        trainer.load_snapshot(snapshot_path=snapshot_path, snapshot_name=snapshot_name)

    trainer._train(max_epochs=max_epochs)




