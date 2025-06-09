from typing import List
from train import main

config={
    'corruption_type': ["gaussian", 'poisson'],
    'corruption_params': {}
}
setup_config = dict()

if __name__ == "__main__":
    main(setup_config, **config)
