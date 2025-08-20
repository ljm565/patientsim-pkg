from .patient import PatientAgent
from .doctor import DoctorAgent

from importlib import resources
__version__ = resources.files("patientsim").joinpath("version.txt").read_text().strip()
