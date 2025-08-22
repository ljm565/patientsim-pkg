from .patient import PatientAgent
from .doctor import DoctorAgent
from .admin_staff import AdminStaffAgent

from importlib import resources
__version__ = resources.files("patientsim").joinpath("version.txt").read_text().strip()
