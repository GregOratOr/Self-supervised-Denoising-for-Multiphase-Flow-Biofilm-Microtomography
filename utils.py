import sys
import numpy as np
from pathlib import Path
import re
from enum import Enum

def compute_spectral_image(image, ifft=False, magnitude=False, normalize=False):
    """
    Computes the spectral image using the magnitude of the 2D Fourier Transform.
    
    Args:
        image (numpy array): A single CT scan patch (grayscale).
    
    Returns:
        spectral_image (numpy array): The spectral representation.
    """
    fft_shift = None
    if ifft:
        fft_shift = np.fft.ifftshift(image)  # Shift zero frequency to center
    else:
        fft_image = np.fft.fft2(image)  # Apply 2D Fourier Transform
        fft_shift = np.fft.fftshift(fft_image)  # Shift zero frequency to center
    
    if magnitude:
        magnitude_spectrum = np.abs(fft_shift)  # Get magnitude
        # Normalize the spectrum for consistency
        magnitude_spectrum = np.log1p(magnitude_spectrum)  # Log scaling
        
        if normalize:
            magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))  # Normalize [0,1]
            
        return fft_shift.astype(np.complex64), magnitude_spectrum
    else: 
        return fft_shift.astype(np.complex64)
    

def get_unique_path(base_path: Path, ext: str = ".pth", pattern_type="_") -> Path:
    """
    Given a base_path (directory + base filename with or without extension),
    return a new Path with an appended suffix _n if needed to avoid overwriting.
    The suffix numbering continues from the highest existing index in the base directory.

    If no files exist with that base name, returns the base_path (with ext fixed if missing).
    If base_path has no extension, `ext` is used by default.

    Parameters:
    - base_path: Path object pointing to file (can have directory and filename)
    - ext: extension to use if base_path has no extension (default ".pth")

    Returns:
    - Path object guaranteed to be unique in base_path.parent directory
    """
    base_dir = base_path.parent
    base_stem = base_path.stem  # filename without suffix or extension

    # Use extension from base_path if present, else default ext
    extension = base_path.suffix if base_path.suffix else ext

    base_dir.mkdir(parents=True, exist_ok=True)

    # Regex pattern: match base_stem, optionally with _number, followed by extension
    if pattern_type == "_":
        pattern = re.compile(rf"^{re.escape(base_stem)}(?:_(\d+))?{re.escape(extension)}$")
    else:
        pattern = re.compile(rf"^{re.escape(base_stem)}(?:\((\d+)\))?{re.escape(extension)}$")

    existing_indices = []
    for f in base_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                if m.group(1) is None:
                    existing_indices.append(0)  # base file without suffix
                else:
                    existing_indices.append(int(m.group(1)))

    if not existing_indices:
        # No prior file, return base_path with ensured extension
        if base_path.suffix == extension:
            return base_path
        else:
            return base_dir / f"{base_stem}{extension}"

    # Find next suffix (start from 1 if base file exists without suffix)
    suffix = max(existing_indices) + 1 if max(existing_indices) > 0 else 1

    unique_name = f"{base_stem}_{suffix}{extension}"
    return base_dir / unique_name

class Logger(object):
        '''
        A Logger class that directs all the calls to standard output stream to a log file.
        '''
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "a", buffering=1)  # Line-buffered

        def __enter__(self):
            sys.stdout = self
            return self

        def __exit__(self, exception_type, exception_value, exception_traceback):
            if self.log:
                self.log.close()
            sys.stdout = sys.__stdout__

        def __del__(self):
            """Ensure log file is closed when Logger object is deleted."""
            if self.log:
                self.log.close()
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def start(self):
            sys.stdout = self
        
        def stop(self):
            if self.log:
                self.log.close()
            sys.stdout = sys.__stdout__

class NoiseType(Enum):
    POISSON = "poisson"
    GAUSSIAN = "gaussian"
    BERNOULLI = "bspec"
    GAUSSIAN_BLUR = "blur"
    NO_NOISE = ""

