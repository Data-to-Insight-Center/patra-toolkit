from abc import ABC, abstractmethod
from typing import Dict
import os
import requests
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
            commit_message="Upload model file"
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
    """
    Handles model storage on GitHub.
    """

    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        raise NotImplementedError("GitHubStore.retrieve_credentials() is not implemented yet.")

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> str:
        raise NotImplementedError("GitHubStore.upload() is not implemented yet.")


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
