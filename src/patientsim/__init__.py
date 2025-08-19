from .patient import PatientAgent


from importlib import resources
__version__ = resources.files("patientsim").joinpath("version.txt").read_text().strip()
