import os
import unittest
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from torchvision import models
from patra_toolkit import ModelCard, AIModel

load_dotenv()


def url_exists(url: str) -> bool:
    """
    Checks if a URL exists.
    Args:
        url (str): URL to check.
    Returns:
        bool: True if the URL exists, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


class TestHuggingFaceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL")
        cls.api = HfApi()
        cls.hf_token = os.environ.get("HF_HUB_TOKEN")

    def setUp(self):
        self.repo_id = "user_model_1.0"

        # Prepare ModelCard
        self.mc = ModelCard(
            name="Model",
            version="1.0",
            short_description="Model",
            full_description="Model",
            keywords="model card",
            author="user",
            input_type="Image",
            category="classification"
        )

        # Attach AIModel to ModelCard
        self.mc.ai_model = AIModel(
            name="Model",
            version="1.0",
            description="Model",
            owner="user",
            location="",
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )

    def test_submit_model(self):
        """
        Test submitting a model to the model store and Patra server.
        """
        model = models.resnet50(pretrained=True)
        model_format = "pt"
        model_store = "huggingface"
        model_location = f"https://huggingface.co/nkarthikeyan/{self.repo_id}/blob/main/{self.repo_id}.pt"

        # Submit the model using ModelCard
        submit_response = self.mc.submit_model(
            patra_server_url=self.patra_server_url,
            model=model,
            model_format=model_format,
            model_store=model_store
        )

        # Assertions
        self.assertIsNotNone(submit_response, "Model submission failed.")
        self.assertTrue(url_exists(model_location), f"Model URL does not exist: {model_location}")

        # Clean up: delete the repository from Hugging Face
        try:
            self.api.delete_repo(repo_id=self.repo_id, token=self.hf_token)
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")

    def test_submit_artifact(self):
        """
        Test submitting an artifact file to the model store and Patra server.
        """
        artifact_file_path = "label.txt"
        artifact_location = f"https://huggingface.co/nkarthikeyan/{self.repo_id}/blob/main/label.txt"

        # Create a temporary artifact file for testing
        try:
            with open(artifact_file_path, "w") as f:
                f.write("label1\nlabel2\nlabel3")

            # Submit the artifact using ModelCard
            submit_response = self.mc.submit_artifact(
                patra_server_url=self.patra_server_url,
                artifact_path=artifact_file_path,
                model_store="huggingface"
            )

            # Assertions
            self.assertIsNotNone(submit_response, "Artifact submission failed.")
            self.assertTrue(url_exists(artifact_location), f"Artifact URL does not exist: {artifact_location}")

        finally:
            # Clean up: delete the repository and remove local file
            try:
                self.api.delete_repo(repo_id=self.repo_id, token=self.hf_token)
                if os.path.exists(artifact_file_path):
                    os.remove(artifact_file_path)
            except Exception as e:
                self.fail(f"Cleanup failed: {e}")


if __name__ == "__main__":
    unittest.main()
