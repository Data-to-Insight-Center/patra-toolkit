import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict

import requests


class ModelStore(ABC):
    """
    Abstract class for a model store that can upload and delete model files.
    """

    @classmethod
    @abstractmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieve credentials for the model store from the Patra server.

        Args:
            patra_server_url (str): URL of the Patra server.
            timeout (int): Timeout in seconds for the request.

        Returns:
            Dict[str, str]: A dictionary containing the credentials

        Raises:
            Exception: If the credentials are missing or invalid
        """
        pass

    @abstractmethod
    def upload(self, file_path: str, pid: str, credentials: Dict[str, str]) -> str:
        """
        Upload a model file to the store.

        Args:
            file_path (str): Path to the model file.
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials for the model store.

        Returns:
            str: URL of the uploaded file

        Raises:
            Exception: If the upload fails
        """
        pass

    @abstractmethod
    def delete_repo(self, pid: str, credentials: Dict[str, str]) -> None:
        """
        Delete a repository or model file from the store.

        Args:
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials for the model store.

        Raises:
            Exception: If the deletion fails
        """
        pass


class HuggingFaceStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieve Hugging Face credentials from the Patra server.

        Args:
            patra_server_url (str): URL of the Patra server.
            timeout (int): Timeout in seconds for the request.

        Returns:
            Dict[str, str]: A dictionary containing the credentials

        Raises:
            Exception: If the credentials are missing or invalid
        """
        response = requests.get(
            f"{patra_server_url}/get_huggingface_credentials",
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        response.raise_for_status()
        creds = response.json()
        if "username" not in creds or "token" not in creds:
            raise Exception("Invalid Hugging Face credentials response from server.")
        return creds

    def upload(self, file_path: str, pid: str, credentials: Dict[str, str]) -> str:
        """
        Upload a model file to the Hugging Face model hub.

        Args:
            file_path (str): Path to the model file.
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials

        Returns:
            str: URL of the uploaded file

        Raises:
            Exception: If the upload fails
        """
        hf_hub = ensure_package_installed("huggingface_hub")  # installs if missing
        from huggingface_hub import create_repo, upload_file

        username, token = credentials["username"], credentials["token"]
        repo_id = f"{username}/{pid.replace(' ', '_')}"
        create_repo(repo_id=repo_id, private=False, exist_ok=True, token=token)
        filename = os.path.basename(file_path)
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
            commit_message="Upload via Patra Toolkit"
        )
        return f"https://huggingface.co/{repo_id}/blob/main/{filename}"

    def delete_repo(self, pid: str, credentials: Dict[str, str]) -> None:
        """
        Delete a repository from the Hugging Face model hub.

        Args:
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials

        Raises:
            Exception: If the deletion fails
        """
        hf_hub = ensure_package_installed("huggingface_hub")  # installs if missing
        from huggingface_hub import HfApi

        username, token = credentials["username"], credentials["token"]
        repo_id = f"{username}/{pid.replace(' ', '_')}"
        api = HfApi()
        api.delete_repo(repo_id=repo_id, token=token)


class GitHubStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieve GitHub credentials from the Patra server.

        Args:
            patra_server_url (str): URL of the Patra server.
            timeout (int): Timeout in seconds for the request.

        Returns:
            Dict[str, str]: A dictionary containing the credentials

        Raises:
            Exception: If the credentials are missing or invalid
        """
        response = requests.get(
            f"{patra_server_url}/get_github_credentials",
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        response.raise_for_status()
        creds = response.json()
        if "username" not in creds or "token" not in creds:
            raise Exception("Invalid GitHub credentials response from server.")
        return creds

    def upload(self, file_path: str, pid: str, credentials: Dict[str, str]) -> str:
        """
        Upload a model file to a GitHub repository.

        Args:
            file_path (str): Path to the model file.
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials

        Returns:
            str: URL of the uploaded file

        Raises:
            Exception: If the upload fails
        """
        ensure_package_installed("PyGithub", "github")
        from github import Github, GithubException

        username, token = credentials["username"], credentials["token"]
        repo_name = pid.replace(' ', '_')
        repo_url = f"https://github.com/{username}/{repo_name}.git"
        github = Github(token)
        try:
            user = github.get_user()
            repo_exists = False
            try:
                repo = user.get_repo(repo_name)
                repo_exists = True
                print(f"Repository '{repo_name}' already exists. Using existing repository.")
            except GithubException:
                repo = user.create_repo(repo_name, private=False)
                print(f"Repository '{repo_name}' created successfully.")
        except GithubException as e:
            raise Exception(f"Failed to create or access GitHub repository: {e}")

        local_dir = tempfile.mkdtemp(prefix=repo_name)
        try:
            subprocess.run(["git", "init"], cwd=local_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=local_dir, check=True)
            if repo_exists:
                subprocess.run(
                    ["git", "pull", "origin", "main", "--allow-unrelated-histories"],
                    cwd=local_dir,
                    check=False
                )
            shutil.copy(file_path, local_dir)
            filename = os.path.basename(file_path)
            subprocess.run(["git", "add", filename], cwd=local_dir, check=True)
            try:
                commit_cmd = subprocess.run(
                    ["git", "commit", "-m", "Upload via Patra Toolkit"],
                    cwd=local_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                if "nothing to commit" in (e.stderr or "").lower() or "nothing to commit" in (e.output or "").lower():
                    print("No changes to commit, skipping commit step.")
                else:
                    raise Exception(f"Git commit failed: {e.stderr or e.output}")

            subprocess.run(["git", "branch", "-M", "main"], cwd=local_dir, check=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=local_dir, check=True)
            return f"https://github.com/{username}/{repo_name}/blob/main/{filename}"
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to upload file to GitHub using git: {e}")
        finally:
            shutil.rmtree(local_dir)

    def delete_repo(self, pid: str, credentials: Dict[str, str]) -> None:
        """
        Delete a GitHub repository.

        Args:
            pid (str): PID of the model card.
            credentials (Dict[str, str]): Credentials

        Raises:
            Exception: If the deletion fails
        """
        ensure_package_installed("PyGithub", "github")
        from github import Github, GithubException

        username, token = credentials["username"], credentials["token"]
        repo_name = pid.replace(' ', '_')
        github = Github(token)
        try:
            repo = github.get_repo(f"{username}/{repo_name}")
            repo.delete()
        except GithubException as e:
            raise Exception(f"Failed to delete GitHub repository: {e}")


def get_model_store(storage_type: str) -> ModelStore:
    """
    Get a model store object based on the storage type.

    Args:
        storage_type (str): Storage type, either 'huggingface' or 'github'.

    Returns:
        ModelStore: A model store object

    Raises:
        ValueError: If the storage type is not supported
    """
    storage_type = storage_type.lower()
    if storage_type == "huggingface":
        return HuggingFaceStore()
    elif storage_type == "github":
        return GitHubStore()
    else:
        raise ValueError(f"Unsupported storage backend: {storage_type}")


def ensure_package_installed(package_name: str, import_name: str = None):
    """
    Ensure that a package is installed and import it.

    Args:
        package_name (str): Name of the package to install.
        import_name (str): Name to use for importing the package.

    Returns:
        module: The imported module

    Raises:
        ImportError: If the package cannot be imported after installation
    """
    import importlib
    import sys
    import subprocess
    import logging

    import_name = import_name or package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        logging.info(f"Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(import_name)
