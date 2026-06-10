from .patient import PatientAgent
from .doctor import DoctorAgent
from .checker import CheckerAgent

from importlib import resources
__version__ = resources.files("patientsim").joinpath("version.txt").read_text().strip()
