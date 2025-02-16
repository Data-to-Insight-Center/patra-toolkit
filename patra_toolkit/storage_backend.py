import os
from abc import ABC, abstractmethod
from huggingface_hub import HfApi, create_repo, upload_file

import requests


# --- StorageBackend Interface ---
class StorageBackend(ABC):
    @abstractmethod
    def upload(self, file_path: str, metadata: dict) -> dict:
        """Uploads the model file and returns metadata (e.g., URL)."""
        pass


# --- HuggingFaceStorage Implementation ---
class HuggingFaceStorage(StorageBackend):
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api = HfApi()

    def upload(self, file_path: str, metadata: dict) -> dict:
        repo_id = f"{self.username}/{metadata['title'].replace(' ', '_')}"
        # Create the repository if it doesn't exist.
        create_repo(repo_id=repo_id, private=False, exist_ok=True, token=self.token)
        # Upload the file to the repository.
        filename = os.path.basename(file_path)
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=self.token,
            commit_message="Upload model file"
        )
        # Return the direct link to the .pt file.
        file_url = f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        return {"url": file_url}

# --- Factory to Select Storage Backend ---
def get_storage_backend(backend_name: str, credentials: dict) -> StorageBackend:
    if backend_name.lower() == "huggingface":
        return HuggingFaceStorage(credentials["username"], credentials["token"])
    elif backend_name.lower() == "github":
        # TODO: Implement GitHubStorage
        pass
    elif backend_name.lower() == "ndp":
        # TODO: Implement NDPStorage
        pass
    else:
        raise ValueError("Unsupported storage backend.")
