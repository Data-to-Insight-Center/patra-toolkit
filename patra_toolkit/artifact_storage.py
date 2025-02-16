import os
from abc import ABC, abstractmethod
from typing import Dict

import requests
from huggingface_hub import create_repo, upload_file


# --- Abstract Base Class: ArtifactStorage ---
class ArtifactStorage(ABC):
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
    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> str:
        """
        Uploads the file to the respective storage backend.

        Args:
            file_path (str): Path to the file to upload.
            metadata (dict): Metadata containing the model name and version.
            patra_server_url (str): URL of the Patra server.

        Returns:
            str: URL of the uploaded file.
        """
        pass


# --- HuggingFaceStorage Implementation ---
class HuggingFaceStorage(ArtifactStorage):
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

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> Dict[str, str]:
        creds = HuggingFaceStorage.retrieve_credentials(patra_server_url)
        username, token = creds["username"], creds["token"]

        repo_id = f"{username}/{metadata['title'].replace(' ', '_')}"
        create_repo(repo_id=repo_id, private=False, exist_ok=True, token=token)

        filename = os.path.basename(file_path)
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
            commit_message="Upload model file"
        )

        return f"https://huggingface.co/{repo_id}/blob/main/{filename}"


# --- GitHubStorage Implementation ---
class GitHubStorage(ArtifactStorage):
    """
    Handles model storage on GitHub.
    """

    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieves GitHub credentials from the Patra server.
        """
        raise NotImplementedError("GitHubStorage.retrieve_credentials() is not implemented yet.")

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> Dict[str, str]:
        """
        Uploads the model file to GitHub.
        """
        raise NotImplementedError("GitHubStorage.upload() is not implemented yet.")


# --- NDPStorage Implementation ---
class NDPStorage(ArtifactStorage):
    """
    Handles model storage on National Data Platform (NDP).
    """

    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        """
        Retrieves NDP credentials from the Patra server.
        """
        raise NotImplementedError("NDPStorage.retrieve_credentials() is not implemented yet.")

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> Dict[str, str]:
        """
        Uploads the model file to NDP.
        """
        raise NotImplementedError("NDPStorage.upload() is not implemented yet.")


# --- Factory to Select Storage Backend ---
def get_artifact_storage(storage_type: str) -> ArtifactStorage:
    """
    Factory function to return the appropriate storage backend.

    Args:
        storage_type (str): The type of storage backend ("huggingface", "github", "ndp").

    Returns:
        ArtifactStorage: An instance of the requested storage backend.
    """
    storage_type = storage_type.lower()
    if storage_type == "huggingface":
        return HuggingFaceStorage()
    elif storage_type == "github":
        return GitHubStorage()
    elif storage_type == "ndp":
        return NDPStorage()
    else:
        raise ValueError(f"Unsupported storage backend: {storage_type}")
