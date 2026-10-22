import numpy as np
import pandas as pd

def readLabData(filedir):
    """
    Reads spectral lab data from a tab-separated file.

    Parameters:
    - filedir (str): Path to the input file.

    Returns:
    - wav (list): List of wavelengths.
    - ref (list): List of reflectance values.
    """
    try:
        lab = pd.read_csv(filedir, sep='\t', header=None)
        if lab.shape[1] < 2:
            raise ValueError("File format error: Expected at least two columns (wavelength, reflectance).")
        
        wav, ref = lab[0].tolist(), lab[1].tolist()
        return wav, ref
    except Exception as e:
        print(f"Error reading file '{filedir}': {e}")
        return None, None


def importspeclib(path):
    """
    Reads and formats spectral data from a .tab file in the PDS system.
    
    Parameters:
    - path (str): Path to the .tab file.

    Returns:
    - wavelength (list): List of wavelengths (converted to microns).
    - reflectance (list): List of reflectance values.
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Ensure the first line is a valid float (total expected entries)
        try:
            total = int(float(lines[0].strip()))
        except ValueError:
            raise ValueError(f"Invalid first line in '{path}'. Expected a number indicating total entries.")
        
        # Ensure enough lines exist in the file
        if len(lines) < total + 1:
            raise ValueError(f"File '{path}' has fewer lines ({len(lines)}) than expected ({total + 1}).")

        # Process the spectral data using NumPy for efficiency
        data = np.genfromtxt(lines[1:total+1], dtype=float)
        if data.shape[1] < 2:
            raise ValueError(f"File '{path}' format error: Expected at least two columns.")

        wavelength = (data[:, 0] * 1e-3).tolist()  # Convert to microns
        reflectance = data[:, 1].tolist()

        return wavelength, reflectance

    except Exception as e:
        print(f"Error processing file '{path}': {e}")
        return None, None
