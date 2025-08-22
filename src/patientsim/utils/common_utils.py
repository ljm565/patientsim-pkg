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



def check_all_patterns_present(text: str) -> bool:
    """
    Check if all patterns for differential diagnosis are present in the text.

    Args:
        text (str): The text to check for patterns.

    Returns:
        bool: True if all patterns are present, False otherwise.
    """
    patterns = [r"1\..*", r"2\..*", r"3\..*", r"4\..*", r"5\..*"]
    return all(re.search(pattern, text) for pattern in patterns)



def detect_ed_termination(text: str) -> bool:
    """
    Detect if the text indicates the end of a conversation or the provision of differential diagnoses in the ED simulation.

    Args:
        text (str): The text to analyze for termination indicators.

    Returns:
        bool: True if termination indicators are found, False otherwise.
    """
    ddx_key = ["ddx ready:", "my top five differential diagnoses", "my top 5", "here are my top", "[ddx]", "[ddx", "here are some potential concerns we need to consider", "following differential diagnoses"]
    all_present = check_all_patterns_present(text)
    end_flag = any(key.lower() in text.lower() for key in ddx_key)
    return all_present or end_flag



def detect_op_termination(text: str) -> bool:
    """
    Detect if the text indicates the end of a conversation or the provision of differential diagnoses in the OP simulation.

    Args:
        text (str): The text to analyze for termination indicators.

    Returns:
        bool: True if termination indicators are found, False otherwise.
    """
    try:
        pattern = re.compile(r'Answer:\s*\d+\.\s*(.+)')
        return bool(pattern.search(text))
    except:
        return False
