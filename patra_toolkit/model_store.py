import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict
import os
import requests
import subprocess
from github import Github, GithubException
from huggingface_hub import create_repo, upload_file, HfApi


class ModelStore(ABC):
    """
    Abstract class for artifact storage backends.
    """

    @classmethod
    @abstractmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieves credentials from the Patra server.

        Args:
            patra_server_url (str): URL of the Patra server.
            timeout (int): Request timeout in seconds.

        Returns:
            dict: Storage credentials.

        Raises:
            Exception: If credentials are invalid or request fails.
        """
        pass

    @abstractmethod
    def upload(self, file_path: str, pid: str, patra_server_url: str) -> str:
        """
        Uploads the file to the respective storage backend.

        Args:
            file_path (str): Path to the file to upload.
            pid (str): Persistent identifier for the model card.
            patra_server_url (str): URL of the Patra server.

        Returns:
            str: URL of the uploaded file.
        """
        pass

    @abstractmethod
    def delete_repo(self, pid: str, patra_server_url: str) -> None:
        """
        Deletes the repository associated with the given persistent identifier (PID).

        Args:
            pid (str): Persistent identifier for the model card.
            patra_server_url (str): URL of the Patra server for credential retrieval.
        """
        pass


class HuggingFaceStore(ModelStore):
    """
    Handles model storage on Hugging Face.
    """

    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        hf_creds_url = f"{patra_server_url}/get_hf_credentials"
        headers = {"Content-Type": "application/json"}
        response = requests.get(hf_creds_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        creds = response.json()
        if "username" not in creds or "token" not in creds:
            raise Exception("Invalid Hugging Face credentials response from server.")
        return creds

    def upload(self, file_path: str, pid: str, patra_server_url: str) -> str:
        """
        Uploads the model file to Hugging Face.

        Args:
            file_path (str): Local path to the file to upload.
            pid (str): Persistent identifier for the model card.
            patra_server_url (str): URL of the Patra server for credential retrieval.

        Returns:
            str: URL of the uploaded file.
        """
        creds = HuggingFaceStore.retrieve_credentials(patra_server_url)
        owner, token = creds["username"], creds["token"]

        # Generate repository name using the provided metadata.
        repo_id = f"{owner}/{pid.replace(' ', '_')}"
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

    def delete_repo(self, pid: str, patra_server_url: str) -> None:
        """
        Deletes the repository associated with the given persistent identifier (PID).

        Args:
            pid (str): Persistent identifier for the model card.
            patra_server_url (str): URL of the Patra server for credential retrieval.

        Raises:
            Exception: If the repository deletion fails.
        """
        creds = HuggingFaceStore.retrieve_credentials(patra_server_url)
        owner, token = creds["username"], creds["token"]

        repo_id = f"{owner}/{pid.replace(' ', '_')}"
        api = HfApi()
        api.delete_repo(repo_id=repo_id, token=token)


class GitHubStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        response = requests.get(f"{patra_server_url}/get_github_credentials",
                                headers={"Content-Type": "application/json"},
                                timeout=timeout)
        response.raise_for_status()
        creds = response.json()

        if "username" not in creds or "token" not in creds:
            raise Exception("Invalid GitHub credentials response from server.")

        return creds

    def upload(self, file_path: str, pid: str, patra_server_url: str) -> str:
        creds = self.retrieve_credentials(patra_server_url)
        username, token = creds["username"], creds["token"]

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
                result = subprocess.run(
                    ["git", "commit", "-m", "Upload via Patra Toolkit"],
                    cwd=local_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                if "nothing to commit" in e.stderr.lower() or "nothing to commit" in e.output.lower():
                    print("No changes to commit, skipping commit step.")
                else:
                    raise Exception(f"Failed to commit file to GitHub using git: {e.stderr or e.output}")

            subprocess.run(["git", "branch", "-M", "main"], cwd=local_dir, check=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=local_dir, check=True)

            return f"https://github.com/{username}/{repo_name}/blob/main/{filename}"
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to upload file to GitHub using git: {e}")
        finally:
            shutil.rmtree(local_dir)

    def delete_repo(self, pid: str, patra_server_url: str) -> None:
        creds = self.retrieve_credentials(patra_server_url)
        username, token = creds["username"], creds["token"]

        repo_name = pid.replace(' ', '_')
        github = Github(token)
        try:
            repo = github.get_repo(f"{username}/{repo_name}")
            repo.delete()
        except GithubException as e:
            raise Exception(f"Failed to delete GitHub repository: {e}")


class NDPStore(ModelStore):
    """
    Handles model storage on National Data Platform (NDP).
    """

    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        raise NotImplementedError("NDPStore.retrieve_credentials() is not implemented yet.")

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> str:
        raise NotImplementedError("NDPStore.upload() is not implemented yet.")


def get_model_store(storage_type: str) -> ModelStore:
    """
    Factory function to return the appropriate model storage backend.

    Args:
        storage_type (str): The type of storage backend ("huggingface", "github", "ndp").

    Returns:
        ModelStore: An instance of the requested storage backend.

    Raises:
        ValueError: If the storage backend type is unsupported.
    """
    storage_type = storage_type.lower()
    if storage_type == "huggingface":
        return HuggingFaceStore()
    elif storage_type == "github":
        return GitHubStore()
    elif storage_type == "ndp":
        return NDPStore()
    else:
        raise ValueError(f"Unsupported storage backend: {storage_type}")
