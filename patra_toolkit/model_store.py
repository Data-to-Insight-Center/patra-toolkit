import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict

import requests
from github import Github, GithubException
from huggingface_hub import create_repo, upload_file, HfApi


class ModelStore(ABC):
    @classmethod
    @abstractmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        pass

    @abstractmethod
    def upload(self, file_path: str, pid: str, credentials: Dict[str, str]) -> str:
        pass

    @abstractmethod
    def delete_repo(self, pid: str, credentials: Dict[str, str]) -> None:
        pass


class HuggingFaceStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        response = requests.get(f"{patra_server_url}/get_huggingface_credentials",
                                headers={"Content-Type": "application/json"}, timeout=timeout)
        response.raise_for_status()
        creds = response.json()
        if "username" not in creds or "token" not in creds:
            raise Exception("Invalid Hugging Face credentials response from server.")
        return creds

    def upload(self, file_path: str, pid: str, credentials: Dict[str, str]) -> str:
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
        username, token = credentials["username"], credentials["token"]
        repo_id = f"{username}/{pid.replace(' ', '_')}"
        api = HfApi()
        api.delete_repo(repo_id=repo_id, token=token)


class GitHubStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
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
                subprocess.run(
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
        username, token = credentials["username"], credentials["token"]
        repo_name = pid.replace(' ', '_')
        github = Github(token)
        try:
            repo = github.get_repo(f"{username}/{repo_name}")
            repo.delete()
        except GithubException as e:
            raise Exception(f"Failed to delete GitHub repository: {e}")


class NDPStore(ModelStore):
    @classmethod
    def retrieve_credentials(cls, patra_server_url: str, timeout: int = 10) -> Dict[str, str]:
        raise NotImplementedError("NDPStore.retrieve_credentials() is not implemented yet.")

    def upload(self, file_path: str, metadata: Dict[str, str], patra_server_url: str) -> str:
        raise NotImplementedError("NDPStore.upload() is not implemented yet.")

    def delete_repo(self, pid: str, credentials: Dict[str, str]) -> None:
        raise NotImplementedError("NDPStore.delete_repo() is not implemented yet.")


def get_model_store(storage_type: str) -> ModelStore:
    storage_type = storage_type.lower()
    if storage_type == "huggingface":
        return HuggingFaceStore()
    elif storage_type == "github":
        return GitHubStore()
    elif storage_type == "ndp":
        return NDPStore()
    else:
        raise ValueError(f"Unsupported storage backend: {storage_type}")
