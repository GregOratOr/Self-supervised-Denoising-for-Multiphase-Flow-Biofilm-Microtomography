import sys
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import date
import re
from enum import Enum

class NoiseType(Enum):
    POISSON = "poisson"
    GAUSSIAN = "gaussian"
    BERNOULLI = "bspec"
    GAUSSIAN_BLUR = "blur"
    NO_NOISE = ""

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
    
def get_unique_path_old(base_path: Path, ext: str="", pattern_type="_") -> Path:
    """
    Given a base_path (directory + base filename with or without extension),
    return a new Path with an appended suffix _n if needed to avoid overwriting.
    The suffix numbering continues from the highest existing index in the base directory.

    If no files exist with that base name, returns the base_path (with ext fixed if missing).
    If base_path has no extension, `ext` is used by default.

    Parameters:
    - base_path: Path object pointing to file (can have directory and filename)
    - ext: extension to use if base_path has no extension (default "")

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

def get_unique_path(base_path: Path, ext: Optional[str]=None, pattern_type: str="_", use_date: bool=False, reuse_empty:bool=True) -> Path:
    '''
    Generate a unique file or directory path by appending a date or index suffix if necessary.

    Parameters:
        base_path (Path): Base file or directory path. \n
        ext (Optional[str]): File extension (e.g., 'txt'). If None, treat as directory. \n 
        pattern_type (str): Suffix style, either '_' (e.g. "_1") or '()' (e.g. "(1)"). \n
        use_date (bool): If True, append today's date to the base name before suffixing. \n

    Returns:
        Path: A unique Path that does not currently exist.
    '''
    today_str = date.today().strftime('%Y-%m-%d')

    # Determine parent directory and base stem
    if base_path.name:
        parent = base_path.parent
        stem = base_path.name
        if use_date:
            stem = f"{stem}{pattern_type if pattern_type in ['_', '-'] else ' '}{today_str}"
    else:
        parent = base_path
        stem = today_str

    # Determine suffix/extension
    suffix = f".{ext}" if ext else ""
    pattern_ext = re.escape(suffix)

    # Build a regex pattern depending on pattern_type
    if pattern_type == "_":
        # Matches: basename or basename_1, basename_2, etc.
        pattern = re.compile(rf"^{re.escape(stem)}(?:_(\d+))?{pattern_ext}$")
        suffix_format = lambda i: f"_{i}"
    elif pattern_type == "-":
        # Matches: basename or basename_1, basename_2, etc.
        pattern = re.compile(rf"^{re.escape(stem)}(?:-(\d+))?{pattern_ext}$")
        suffix_format = lambda i: f"-{i}"
    elif pattern_type == "()":
        # Matches: basename or basename(1), basename(2), etc.
        pattern = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?{pattern_ext}$")
        suffix_format = lambda i: f"({i})"
        # When adding date before, add '_' since '(date)' looks odd for date
        if use_date and base_path.name:
            stem = f"{stem}_{today_str}"
            pattern = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?{pattern_ext}$")
    elif pattern_type == "[]":
        # Matches: basename or basename[1], basename[2], etc.
        pattern = re.compile(rf"^{re.escape(stem)}(?:\[(\d+)\])?{pattern_ext}$")
        suffix_format = lambda i: f"({i})"
        # When adding date before, add '_' since '(date)' looks odd for date
        if use_date and base_path.name:
            stem = f"{stem}_{today_str}"
            pattern = re.compile(rf"^{re.escape(stem)}(?:\[(\d+)\])?{pattern_ext}$")
    else:
        raise ValueError(f"Unsupported pattern_type '{pattern_type}', use '_' or '()'")

    # Scan existing files/folders to find used indices
    existing_indices = []
    for p in parent.iterdir():
        if ext:
            if not p.is_file() or not p.name.endswith(suffix):
                continue
        else:
            if not p.is_dir():
                continue
        match = pattern.match(p.name)
        if match:
            # No suffix counts as index 0
            existing_indices.append(int(match.group(1)) if match.group(1) else 0)

    # Candidate unindexed path
    target_path = parent / f"{stem}{suffix}"
    if not target_path.exists():
        return target_path
    
    # If exists, find next index
    if reuse_empty:
        next_index = max(existing_indices) if existing_indices else 1
        temp = parent / f"{stem}{suffix_format(next_index)}{suffix}"
        if temp.is_dir() and not any(temp.iterdir()):
            return temp
        
    next_index = max(existing_indices) + 1 if existing_indices else 1
    unique_name = f"{stem}{suffix_format(next_index)}{suffix}"
    return parent / unique_name

def get_unique_run_path(base_path: Path) -> Path:
    """
    Generate a unique directory path of the format: YYYY-MM-DD-stem-0, YYYY-MM-DD-stem-1, etc.

    Parameters:
        base_path (Path): Base directory path; if no name is provided, uses 'run' as default stem.

    Returns:
        Path: A unique, non-existent Path object.
    """
    today_str = date.today().strftime('%Y-%m-%d')
    stem ="run"
    prefix = f"{today_str}_{stem}"

    base_path.mkdir(parents=True, exist_ok=True)
    
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")

    existing_indices = [
        int(m.group(1))
        for p in base_path.iterdir() if p.is_dir()
        for m in [pattern.match(p.name)] if m
    ]

    next_index = max(existing_indices) + 1 if existing_indices else 0
    unique_name = f"{prefix}-{next_index}"
    return base_path / unique_name

class Logger:
    """
    A flexible Logger class that logs messages to a file, optionally printing to the terminal.
    """
    def __init__(self, filepath, init_log: bool = False):
        self.log_file_path = Path(filepath)
        self.log_file = None
        self.logging_active = False

        if self.log_file_path.suffix == "":
            self.log_file_path = self.log_file_path.with_suffix(".txt")

        if init_log:
            self._open_log_file()

    def _open_log_file(self):
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, "a", buffering=1)  # line-buffered

    def start(self):
        """
        Start logging to the file.
        """
        if not self.log_file:
            self._open_log_file()
        self.logging_active = True

    def pause(self):
        """
        Stop logging (does not close the file).
        """
        self.logging_active = False

    def resume(self):
        """
        Resume logging
        """
        self.logging_active = True
        
    def stop(self):
        """
        Stop logging and close  the log file.
        """
        self.logging_active = False
        self.close()

    def close(self):
        """
        Close the log file explicitly.
        """
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.logging_active = False

    def log(self, msg: str, terminal: bool = True):
        """
        Log a message to file and optionally print to terminal.
        """
        if terminal:
            print(msg)  # Avoid double newline since print adds one

        if self.logging_active and self.log_file:
            self.log_file.write(msg)
            if not msg.endswith("\n"):
                self.log_file.write("\n")
            self.log_file.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


