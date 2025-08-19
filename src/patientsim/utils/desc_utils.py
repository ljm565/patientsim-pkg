import random

from patientsim.registry.term import *
from patientsim.registry.persona import *
from patientsim.utils import colorstr



def get_personality_description(personality: str) -> str:
    """
    Generate a description for the patient's personality type.

    Args:
        personality (str): The personality type (e.g., plain, verbose, pleasing, etc.).

    Returns:
        str: A formatted description of the patient's personality type.
    """
    indent = "\n\t\t"
    lines = PERSONALITY[personality]["prompt"].split("\n")
    description = indent + indent.join(lines)
    if not personality == "plain":
        description += (
            f"{indent}IMPORTANT: Ensure that your personality is clearly represented "
            "throughout the conversation, while allowing your emotional tone and style to vary naturally across turns.")
    return description



def get_recall_description(level: str) -> str:
    """
    Generate a description for the patient's recall level.

    Args:
        level (str): The recall level (no_history, low, or high).

    Returns:
        str: A formatted description of the patient's recall level.
    """
    indent = "\n\t\t"
    lines = RECALL_LEVEL[level]["prompt"].split("\n")
    return f"{level.capitalize()}{indent}" + indent.join(lines)



def get_confusion_description(level: str) -> str:
    """
    Generate a description for the patient's confusion level.

    Args:
        level (str): The confusion level (high, moderate, or normal).

    Returns:
        str: A formatted description of the patient's confusion level.
    """
    indent = "\n\t\t"
    if level == "normal":
        lines = CONFUSION_LEVEL[level]["prompt"].split("\n")
        return f"{level.capitalize()}{indent}" + indent.join(lines)
    
    description = (
        f"\n\tThe patient's initial dazed level is {level}. "
        + "The dazedness should gradually fade throughout the conversation as the doctor continues to reassure them. "
        + "Transitions should feel smooth and natural, rather than abrupt. "
        + "While the change should be subtle and progressive, the overall dazed level is expected to decrease noticeably every 4-5 turns, following the instructions for each level below."
    )
    
    for level_to_include in CONFUSION_TO_INCLUDE[level]:
        description += f"\n\t{level_to_include.capitalize()} Dazedness ({CONFUSION_STATE[level_to_include].capitalize()} Phase){indent}" + indent.join(
            CONFUSION_LEVEL[level_to_include]["prompt"].split("\n")
        )

    description += "\n\tNote: Dazedness reflects the patient's state of confusion and inability in following the conversation, independent of their language proficiency."

    return description



def get_language_proficiency_description(level: str,
                                         num_sample: int, 
                                         random_sampling: bool = True, 
                                         **kwargs) -> str:
    """
    Generate a description for the patient's language proficiency level.

    Args:
        level (str): The language proficiency level (A, B, or C).
        num_sample (int): The number of words to sample from the vocabulary.
        random_sampling (bool, optional): Whether to randomly sample words from the vocabulary. Defaults to True.

    Raises:
        ValueError: If the provided level is not one of 'A', 'B', or 'C'.

    Returns:
        str: A formatted description of the patient's language proficiency level.
    """
    if level == "A":
        understand_words = kwargs.get('cefr_A1') if kwargs.get('cefr_A1') else CEFR_A1
        misunderstand_words = kwargs.get('cefr_A2') if kwargs.get('cefr_A2') else CEFR_A2
        understand_med_words = kwargs.get('med_A') if kwargs.get('med_A') else MED_A
        misunderstand_med_words = kwargs.get('med_B') if kwargs.get('med_B') else MED_B
    elif level == "B":
        understand_words = kwargs.get('cefr_B1') if kwargs.get('cefr_B1') else CEFR_B1
        misunderstand_words = kwargs.get('cefr_B2') if kwargs.get('cefr_B2') else CEFR_B2
        understand_med_words = kwargs.get('med_B') if kwargs.get('med_B') else MED_B
        misunderstand_med_words = kwargs.get('med_C') if kwargs.get('med_C') else MED_C
    elif level == "C":
        understand_words = kwargs.get('cefr_C1') if kwargs.get('cefr_C1') else CEFR_C1
        misunderstand_words = kwargs.get('cefr_C2') if kwargs.get('cefr_C2') else CEFR_C2
        understand_med_words = kwargs.get('med_C') if kwargs.get('med_C') else MED_C
        misunderstand_med_words = []
    else:
        raise ValueError(colorstr("red", f"Invalid language proficiency level: {level}. Must be one of 'A', 'B', or 'C'."))
    
    understand_words = random.sample(understand_words, min(len(understand_words), num_sample)) if random_sampling else understand_words[:num_sample]
    misunderstand_words = random.sample(misunderstand_words, min(len(misunderstand_words), num_sample)) if random_sampling else misunderstand_words[:num_sample]
    understand_med_words = random.sample(understand_med_words, min(len(understand_med_words), num_sample)) if random_sampling else understand_med_words[:num_sample]
    misunderstand_med_words = random.sample(misunderstand_med_words, min(len(misunderstand_med_words), num_sample)) if random_sampling else misunderstand_med_words[:num_sample]
    words_type = {
        "understand_words": ", ".join(understand_words),
        "misunderstand_words": ", ".join(misunderstand_words),
        "understand_med_words": ", ".join(understand_med_words),
        "misunderstand_med_words": ", ".join(misunderstand_med_words) if misunderstand_med_words else ""
    }

    # Generate the description based on the level
    indent = "\n\t\t\t"
    lines = LANG_PROFICIENCY_LEVEL[level]["prompt"].split("\n")
    description = "\n\t\t" + indent.join(lines).format(**words_type)

    return description



def get_reminder_description(personality: str, 
                             lang_proficiency_level: str, 
                             recall_level: str, 
                             confusion_level: str) -> str:
    """
    Generate a reminder description for the patient based on their attributes.

    Args:
        personality (str): The personality type of the patient (e.g., plain, verbose, etc.).
        lang_proficiency_level (str): The language proficiency level of the patient (e.g., A, B, C).
        recall_level (str): The recall level of the patient (e.g., no_history, low, high).
        confusion_level (str): The confusion level of the patient (e.g., high, moderate, normal).

    Returns:
        str: A formatted reminder description for the patient.
    """

    description_list = [
        f"You should act like {LANG_PROFICIENCY_LEVEL[lang_proficiency_level]['description'].lower()}",
        f"You are {PERSONALITY[personality]['description'].lower()}",
        f"Also, you {RECALL_LEVEL[recall_level]['description'].lower()}",
        CONFUSION_LEVEL[confusion_level]['description']
    ]
    return " ".join(description_list)