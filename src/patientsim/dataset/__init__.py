from .manager import DatasetManager

from importlib import resources
__version__ = resources.files("patientsim").joinpath("version.txt").read_text().strip()
