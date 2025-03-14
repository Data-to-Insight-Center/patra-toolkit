import os
import unittest
import requests
from dotenv import load_dotenv
from torchvision import models
from patra_toolkit import ModelCard, AIModel
from patra_toolkit.model_store import get_model_store

load_dotenv()

def url_exists(url: str) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False

class TestGitHubStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL", "http://127.0.0.1:5002")
        cls.github_token = os.environ.get("GITHUB_TOKEN")
        cls.github_username = os.environ.get("GITHUB_USERNAME", "test-user")
        cls.github_store = get_model_store("github")

    def setUp(self):
        self.repo_id = "user_model_0.1"
        self.mc = ModelCard(
            name="GitHubTest",
            version="0.1",
            short_description="Testing GitHub store integration",
            full_description="A test that uploads model artifacts to GitHub, then cleans up.",
            keywords="github, test",
            author="user",
            input_type="Image",
            category="classification"
        )
        self.mc.ai_model = AIModel(
            name="GitHubModel",
            version="0.1",
            description="Testing GitHub integration",
            owner="user",
            location="",
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.99
        )

    def test_submit_model(self):
        model = models.resnet18(pretrained=True)
        resp = self.mc.submit_model(
            patra_server_url=self.patra_server_url,
            model=model,
            file_format="pt",
            model_store="github"
        )
        self.assertIsNotNone(resp)
        self.assertNotIn("error", resp, f"submit_model error: {resp.get('error')}")
        final_model_url = self.mc.ai_model.location
        self.assertTrue(url_exists(final_model_url), f"Model URL not found: {final_model_url}")

        try:
            username = self.mc.credentials["username"]
            repo_name = self.mc.id.replace(" ", "_")
            self.github_store.delete_repo(pid=repo_name, credentials=self.mc.credentials)
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")

    def test_submit_artifact(self):
        model = models.resnet18(pretrained=True)
        resp = self.mc.submit_model(
            patra_server_url=self.patra_server_url,
            model=model,
            file_format="pt",
            model_store="github"
        )
        self.assertNotIn("error", resp, f"submit_model error: {resp.get('error')}")

        artifact_path = "labels_github.txt"
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write("label1\nlabel2\nlabel3")

        try:
            artifact_resp = self.mc.submit_artifact(artifact_path)
            self.assertIsNotNone(artifact_resp)
            self.assertNotIn("error", artifact_resp, f"submit_artifact error: {artifact_resp.get('error')}")
            base_url = self.mc.ai_model.location.rsplit('/', 1)[0]
            artifact_url = f"{base_url}/labels_github.txt"
            self.assertTrue(url_exists(artifact_url), f"Artifact URL not found: {artifact_url}")
        finally:
            try:
                username = self.mc.credentials["username"]
                repo_name = self.mc.id.replace(" ", "_")
                if os.path.exists(artifact_path):
                    os.remove(artifact_path)
                self.github_store.delete_repo(pid=repo_name, credentials=self.mc.credentials)
            except Exception as e:
                self.fail(f"Cleanup failed: {e}")

if __name__ == "__main__":
    unittest.main()
