from typing import List
from core.train import launch, main
from core.utils import get_unique_run_path, NoiseType
from easydict import EasyDict
from pathlib import Path
from core.dataset import CTScans

config = EasyDict()
setup_config = EasyDict()
run_desc = ""

####################
#################### Setup Config Hyperparameters ####################
# Experiment
experiment_id = 55
experiment_desc = "Full_seq_py"

# Training Hyperparameters
results_dir = Path("../results")
learning_rate = 0.0005
max_epochs = 250
batch_size = 32
lr_scheduler_type = 'constant' # 'constant' or 'plateau'
load_checkpoint = False
snapshot_name = "latest-snapshot"
snapshot_path = results_dir / "055-Python-script-test/2025-06-11_run-1"

# Post Processing
post_operation = 'fspec'

# Dataset paths
dataset_root_path = Path("../datasets")
train_dataset = f"HQ_train_dataset_{100}_patch_{str(512)}_olap_{str(0.2)}.pth" # "HQ_eval_dataset_10.pth"
valid_dataset = f"HQ_valid_dataset_{20}_patch_{str(512)}_olap_{str(0.2)}.pth" # "HQ_valid_dataset_5.pth"

#################### Run Parameters ####################
# Noise2Noise or Noise2Clean (Default: Noise2Noise)
n2n = True

# Augmentation Hyperparameters
augment_params = {'translate': 64}

# Corruption Hyperparameters
corruption_types = [ NoiseType.POISSON, NoiseType.GAUSSIAN, NoiseType.BERNOULLI, NoiseType.GAUSSIAN_BLUR ]
persistant_noise = False # uses same noise for all patches.
target_noise_injection_factor = 0.7

# Gaussian Params
gaussian_params = {'std': 50}

# Poisson Params
poisson_params = {
    'strength': 50,
    'distribution': 'uniform', # uniform or gaussian
    'mask_ratio': 1.0,
    'sigma': 1.0
}

# Bernoulli Params
bernoulli_params = {'p_edge': 0.95}

# Blur Params
blur_params = {'sigma': 1.0, 'kernel': 15}

#################### Setup Update Config ####################
setup_config.experiment_id = experiment_id
setup_config.experiment_desc = experiment_desc

setup_config.lr = learning_rate
setup_config.max_epochs = max_epochs
setup_config.batch_size = batch_size
setup_config.lr_scheduler_type = lr_scheduler_type
setup_config.load_checkpoint = load_checkpoint

if load_checkpoint:
    setup_config.results_dir = snapshot_path
    setup_config.snapshot_name = snapshot_name
    setup_config.snapshot_path = snapshot_path / "Snapshots"
else:
    if experiment_desc == '':
        new_pth = results_dir / f"{experiment_id:03d}"
        setup_config.results_dir = Path(get_unique_run_path(new_pth))
    else:
        setup_config.results_dir = Path(get_unique_run_path(results_dir / f"{experiment_id:03d}-{experiment_desc}"))

setup_config.post_op = post_operation

setup_config.train_dataset_path = dataset_root_path/train_dataset
setup_config.valid_dataset_path = dataset_root_path/valid_dataset

#################### Update Config ####################
corruption_params = {}
config.n2n = n2n
config.augment_params = augment_params
config.corruption_types = corruption_types
corruption_params.update(gaussian_params=gaussian_params, 
                         poisson_params=poisson_params, 
                         bspec_params=bernoulli_params, 
                         blur_params=blur_params, 
                         target_noise_injection_factor=target_noise_injection_factor,
                         persistant_noise=persistant_noise)
config.corruption_params = corruption_params

#################### Build Run Description ####################
# Experiment
run_desc += f'Exp-[{experiment_id:03d}]'
run_desc += f'-{experiment_desc}'
run_desc += f'-[Train]: '
# Training Hyperparameters
run_desc += f'n_eps-[{max_epochs}]-batch-[{batch_size}]-lr-[{learning_rate}]-lr_sch-[{lr_scheduler_type}]'
# Load checkpoint
run_desc += f'LOAD-[{snapshot_name}]-' if load_checkpoint else ''
# Post-Processing
run_desc += post_operation + '-' if post_operation else ''
# Datasets
run_desc += f'Train[{train_dataset}]-Valid[{valid_dataset}]-'


# Noise2Noise / Noise2Clean
run_desc += "n2n-" if n2n else "n2c-"
# Augmentation
run_desc += f'translate-[{augment_params.get("translate", "")}]-' if augment_params.get('translate', None) else ''
# Corruption Params
run_desc += f"[{', '.join(f'{k}={v}' for k, v in corruption_params.items())}]"

#################### Main ####################
if __name__ == "__main__":
    # print(run_desc)
    # print(config)

    # Old 1-GPU version
    # main(setup_config, config)

    launch(setup_config, config)
