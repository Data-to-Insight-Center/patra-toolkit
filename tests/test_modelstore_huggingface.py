import os
import unittest
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from torchvision import models

from patra_toolkit import ModelCard, AIModel

load_dotenv()


def url_exists(url: str) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


class TestHuggingFaceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = os.getenv("PATRA_SERVER_URL", "http://127.0.0.1:5002")
        cls.api = HfApi()
        cls.hf_token = os.getenv("HF_HUB_TOKEN")

    def setUp(self):
        self.repo_id = "test_resnet_model_1.0"

        # Create ModelCard
        self.mc = ModelCard(
            name="ResNetTest",
            version="1.0",
            short_description="A test ResNet model",
            full_description="Testing ResNet model submission",
            keywords="resnet, test",
            author="test-user",
            input_type="Image",
            category="classification"
        )

        # Attach AIModel to ModelCard
        self.mc.ai_model = AIModel(
            name="TestResNet",
            version="1.0",
            description="Sample ResNet model for testing",
            owner="test-user",
            location="",
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )

    def test_submit_model(self):
        model = models.resnet50(pretrained=True)
        resp = self.mc.submit(patra_server_url=self.patra_server_url, model=model, file_format="pt",
                              model_store="huggingface")
        self.assertIsNotNone(resp, "submit_model returned None")
        self.assertNotIn("error", resp, f"submit_model error: {resp.get('error')}")

        final_model_url = self.mc.ai_model.location
        self.assertIsNotNone(final_model_url, "No model location found in AIModel")

        self.assertTrue(
            url_exists(final_model_url),
            f"Model URL does not exist or is invalid: {final_model_url}"
        )

        try:
            username = self.mc.credentials["username"]
            repo_name = self.mc.id.replace(" ", "_")
            repo_id = f"{username}/{repo_name}"
            self.api.delete_repo(repo_id=repo_id, token=self.hf_token)
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")

    def test_submit_artifact(self):
        model = models.resnet50(pretrained=True)
        resp = self.mc.submit(patra_server_url=self.patra_server_url, model=model, file_format="pt",
                              model_store="huggingface")
        self.assertNotIn("error", resp, f"submit_model error: {resp.get('error')}")

        artifact_file_path = "artifact.txt"
        with open(artifact_file_path, "w") as f:
            f.write("test artifact data")

        try:
            artifact_resp = self.mc.submit_artifact(artifact_file_path)
            self.assertIsNotNone(artifact_resp, "submit_artifact returned None")
            self.assertNotIn("error", artifact_resp, f"submit_artifact error: {artifact_resp.get('error')}")

            final_model_url = self.mc.ai_model.location
            artifact_url_base = final_model_url.rsplit('/', 1)[0]
            final_artifact_url = f"{artifact_url_base}/artifact.txt"

            self.assertTrue(
                url_exists(final_artifact_url),
                f"Artifact URL not found or invalid: {final_artifact_url}"
            )
        finally:
            try:
                username = self.mc.credentials["username"]
                repo_name = self.mc.id.replace(" ", "_")
                repo_id = f"{username}/{repo_name}"
                if os.path.exists(artifact_file_path):
                    os.remove(artifact_file_path)
                self.api.delete_repo(repo_id=repo_id, token=self.hf_token)
            except Exception as e:
                self.fail(f"Cleanup failed: {e}")


if __name__ == "__main__":
    unittest.main()
