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

# --- GitHubStorage Implementation ---
class GitHubStorage(StorageBackend):
    def __init__(self, username: str, token: str, repo_name: str = "patra-models"):
        self.username = username
        self.token = token
        self.repo_name = repo_name
        self.github_api = f"https://api.github.com/repos/{self.username}/{self.repo_name}"
        self._ensure_repo_exists()

    def _ensure_repo_exists(self):
        headers = {"Authorization": f"token {self.token}"}
        response = requests.get(self.github_api, headers=headers)
        if response.status_code == 404:
            create_repo_url = "https://api.github.com/user/repos"
            payload = {"name": self.repo_name, "private": False}
            response = requests.post(create_repo_url, json=payload, headers=headers)
            if response.status_code not in (200, 201):
                raise Exception("Failed to create GitHub repository.")

    def upload(self, file_path: str, metadata: dict) -> dict:
        tag_name = "v1.0.0"
        release_url = f"{self.github_api}/releases"
        headers = {"Authorization": f"token {self.token}"}
        release_data = {"tag_name": tag_name, "name": metadata["title"], "body": "Model upload"}
        release_response = requests.post(release_url, json=release_data, headers=headers)
        if release_response.status_code not in (200, 201):
            raise Exception("Failed to create GitHub release.")
        release = release_response.json()
        upload_url = release["upload_url"].replace("{?name,label}", "")
        with open(file_path, "rb") as f:
            upload_response = requests.post(
                upload_url,
                params={"name": os.path.basename(file_path)},
                data=f,
                headers={"Authorization": f"token {self.token}", "Content-Type": "application/octet-stream"}
            )
        if upload_response.status_code not in (200, 201):
            raise Exception("Failed to upload file to GitHub release.")
        return {"url": release["html_url"]}


# --- NDPStorage Implementation ---
class NDPStorage(StorageBackend):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload(self, file_path: str, metadata: dict) -> dict:
        # Hypothetical NDP upload endpoint
        NDP_UPLOAD_URL = "https://ndp.example.edu/api/upload"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            response = requests.post(NDP_UPLOAD_URL, files=files, headers=self.headers)
        if response.status_code not in (200, 201):
            raise Exception("Failed to upload file to NDP.")
        return {"url": response.json().get("url", "")}


# --- Factory to Select Storage Backend ---
def get_storage_backend(backend_name: str, credentials: dict) -> StorageBackend:
    if backend_name.lower() == "huggingface":
        return HuggingFaceStorage(credentials["username"], credentials["token"])
    elif backend_name.lower() == "github":
        return GitHubStorage(credentials["username"], credentials["token"])
    elif backend_name.lower() == "ndp":
        return NDPStorage(credentials["api_key"])
    else:
        raise ValueError("Unsupported storage backend.")