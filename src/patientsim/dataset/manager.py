import shutil
import getpass
import subprocess
from pathlib import Path
from typing import Optional, Literal

from patientsim.utils import log



class DatasetManager:
    def __init__(self, save_path: str) -> None:
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)


    def download(self, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None, 
                 dataset_version: str = "1.0.0", 
                 mode: Literal["all", "profile"] = "profile") -> None:
        """
        Download the dataset from Physionet and save it to the specified path.

        Args:
            username (Optional[str]): Physionet username. If None, prompts for input.
            password (Optional[str]): Physionet password. If None, prompts for input.
            dataset_version (str): Version of the dataset to download. Default is "1.0.0".
            mode (Literal["all", "profile"]): Download mode.
                - "all": Download entire dataset
                - "profile": Download only patient_profile.json
        """
        # Get credentials
        if username is None:
            username = input("Physionet Username: ")
        if password is None:
            password = getpass.getpass("Physionet Password: ")

        if mode == "all":
            self._download_all(username, password, dataset_version)
        elif mode == "profile":
            self._download_profile(username, password, dataset_version)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'all' or 'profile'")


    def _download_all(self, username: str, password: str, dataset_version: str) -> None:
        """Download entire dataset using wget"""
        url = f"https://physionet.org/files/persona-patientsim/{dataset_version}/"

        # Temporary download directory
        temp_dir = self.save_path / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        log(f"Downloading entire dataset from {url}...")
        log(f"This may take a while...")

        try:
            # Download using wget
            cmd = [
                "wget",
                "-r",  # recursive
                "-N",  # timestamping
                "-c",  # continue
                "-np",  # no parent
                f"--user={username}",
                f"--password={password}",
                "-P",
                str(temp_dir),  # save to temp directory
                url,
            ]

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            log("Download completed successfully!")

            # Move files from nested directory to target path
            self._move_files(temp_dir, dataset_version)

        except subprocess.CalledProcessError as e:
            log(f"Error during download: {e}", level='error')
            log(f"stderr: {e.stderr}", level='error')
            raise
        except Exception as e:
            log(f"Unexpected error: {e}", level='error')
            raise
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                log("Cleaned up temporary files.", level='error')


    def _download_profile(self, username: str, password: str, dataset_version: str) -> None:
        """Download only patient_profile.json"""
        file_url = f"https://physionet.org/files/persona-patientsim/{dataset_version}/patient_profile.json"
        file_path = self.save_path / "patient_profile.json"

        log(f"Downloading patient_profile.json from {file_url}...")

        try:
            cmd = [
                "wget",
                f"--user={username}",
                f"--password={password}",
                "-O", str(file_path),  # output file
                file_url
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            log(f"Successfully downloaded patient_profile.json to {file_path}")
            
        except subprocess.CalledProcessError as e:
            log(f"Error during download: {e}", level='error')
            log(f"stderr: {e.stderr}", level='error')
            if "403" in e.stderr or "401" in e.stderr:
                log("\nAuthentication failed. Please check your credentials.", level='error')
                log("Make sure you have access to this dataset on Physionet.", level='error')
            raise
        except FileNotFoundError:
            log("wget is not installed. Please install wget or use mode='all' with full download.", level='error')
            raise
        except Exception as e:
            log(f"Unexpected error: {e}", level='error')
            raise
        

    def _move_files(self, temp_dir: Path, dataset_version: str) -> None:
        """
        Move downloaded files from nested physionet directory to target path.

        Args:
            temp_dir (Path): Temporary download directory
            dataset_version (str): Version of the dataset
        """
        # Source path: temp_dir/physionet.org/files/persona-patientsim/1.0.0/
        source_path = temp_dir / "physionet.org" / "files" / "persona-patientsim" / dataset_version

        if not source_path.exists():
            raise FileNotFoundError(f"Downloaded files not found at {source_path}")

        log(f"Moving files from {source_path} to {self.save_path}...")

        # Move all files and subdirectories
        for item in source_path.iterdir():
            dest = self.save_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        log(f"Files successfully moved to {self.save_path}")
