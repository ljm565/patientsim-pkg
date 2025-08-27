import re
import random
import numpy as np
from typing import Union
from datetime import datetime, timedelta

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


def str_to_datetime(iso_time: Union[str, datetime]) -> datetime:
    """
    Convert a string representation of a date/time to a datetime object, or return the input if it's already a datetime/date object.

    Args:
        iso_time (Union[str, datetime]): The input date/time, either as a string in the specified format or as a datetime/date object.

    Returns:
        datetime: A datetime object corresponding to the input string, or the original datetime/date if already provided.

    Raises:
        ValueError: If `iso_time` is not a string or datetime/date object.
    """
    try:
        if isinstance(iso_time, str): 
            return datetime.fromisoformat(iso_time)
        return iso_time
    except:
        raise ValueError(colorstr("red", f"`iso_time` must be str or date format, but got {type(iso_time)}"))


def datetime_to_str(iso_time: Union[str, datetime], format: str) -> str:
    """
    Convert a datetime object to a formatted string, or return the input if it is already a string.

    Args:
        iso_time (Union[str, datetime]): The input value to convert. If a datetime object, it will be formatted as a string. 
                                         If already a string, it will be returned as-is.
        format (str): The format string used to convert the datetime object to a string 
                      (e.g., "%Y-%m-%d" or "%Y-%m-%dT%H:%M:%S").

    Returns:
        str: The formatted string representation of the datetime, or the original string if input was a string.

    Raises:
        ValueError: If `iso_time` is neither a string nor a datetime object.
    """
    try:
        if not isinstance(iso_time, str): 
            return iso_time.strftime(format)
        return iso_time
    except:
        raise ValueError(colorstr("red", f"`iso_time` must be str or date format, but got {type(iso_time)}"))


def generate_random_date(start_date: Union[str, datetime] = '1960-01-01',
                         end_date: Union[str, datetime] = '2000-12-31') -> str:
    """
    Generate a random date string in 'YYYY-MM-DD' format between the given start and end dates.

    Args:
        start_date (Union[str, datetime]): The start date in 'YYYY-MM-DD' format. Default is '2000-01-01'.
        end_date (Union[str, datetime]): The end date in 'YYYY-MM-DD' format. Default is '2025-12-31'.

    Returns:
        str: A randomly generated date string in 'YYYY-MM-DD' format.
    """
    start = str_to_datetime(start_date)
    end = str_to_datetime(end_date)
    delta = (end - start).days
    random_days = random.randint(0, delta)
    random_date = start + timedelta(days=random_days)
    return datetime_to_str(random_date, '%Y-%m-%d')


def exponential_backoff(retry_count: int,
                        base_delay: int = 5,
                        max_delay: int = 65,
                        jitter: bool = True) -> float:
    """
    Exponential backoff function for API calling.

    Args:
        retry_count (int): Retry count.
        base_delay (int, optional): Base delay seconds. Defaults to 5.
        max_delay (int, optional): Maximum delay seconds. Defaults to 165.
        jitter (bool, optional): Whether apply randomness. Defaults to True.

    Returns:
        float: Final delay time.
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    if jitter:
        delay = random.uniform(delay * 0.8, delay * 1.2)
    return delay
