import re
import random
import numpy as np
from typing import Union

import torch

from patientsim.utils import colorstr



def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def split_string(string: Union[str, list], delimiter: str = ",") -> list:
    """
    Split a string or list of strings into a list of substrings.

    Args:
        string (Union[str, list]): The string or list of strings to split.
        delimiter (str): The delimiter to use for splitting.

    Returns:
        list: A list of substrings.
    """
    if isinstance(string, str):
        return [s.strip() for s in string.split(delimiter)]
    elif isinstance(string, list):
        return [s.strip() for s in string]
    else:
        raise ValueError(colorstr("red", "Input must be a string or a list of strings."))
    


def prompt_valid_check(prompt: str, data_dict: dict) -> None:
    """
    Check if all keys in the prompt are present in the data dictionary.

    Args:
        prompt (str): The prompt string containing placeholders for data.
        data_dict (dict): A dictionary containing data to fill in the prompt.

    Raises:
        ValueError: If any keys in the prompt are not found in the data dictionary.
    """
    keys = re.findall(r'\{(.*?)\}', prompt)
    missing_keys = [key for key in keys if key not in data_dict]
    
    if missing_keys:
        raise ValueError(colorstr("red", f"Missing keys in the prompt: {missing_keys}. Please ensure all required keys are present in the data dictionary."))
    