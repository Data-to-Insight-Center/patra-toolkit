import os
import unittest
import requests
from dotenv import load_dotenv
from torchvision import models
from patra_toolkit import ModelCard, AIModel
from patra_toolkit.model_store import get_model_store

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


class TestGitHubStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL")
        cls.github_token = os.environ.get("GITHUB_TOKEN")
        cls.github_username = os.environ.get("GITHUB_USERNAME")
        cls.github_store = get_model_store("github")

    def setUp(self):
        self.repo_id = "user_model_0.1"
        self.mc = ModelCard(
            name="Model",
            version="0.1",
            short_description="Testing GitHub store integration",
            full_description="A test that uploads model artifacts to GitHub, then cleans up.",
            keywords="github, test",
            author="user",
            input_type="Image",
            category="classification"
        )

        self.mc.ai_model = AIModel(
            name="Model",
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
        """
        Test submitting a model to the model store and Patra server.
        """
        model = models.resnet18(pretrained=True)
        model_format = "pt"
        model_store = "github"

        submit_response = self.mc.submit_model(
            patra_server_url=self.patra_server_url,
            model=model,
            model_format=model_format,
            model_store=model_store
        )

        self.assertIsNotNone(submit_response, "Model submission failed.")
        self.assertTrue(url_exists(self.mc.output_data), f"Model URL does not exist: {self.mc.output_data}")

    def test_submit_artifact(self):
        """
        Test submitting an artifact file to the model store and Patra server.
        """
        artifact_file_path = "labels_github.txt"
        artifact_store = "github"

        with open(artifact_file_path, "w", encoding="utf-8") as f:
            f.write("label1\nlabel2\nlabel3")

        submit_response = self.mc.submit_artifact(
            patra_server_url=self.patra_server_url,
            artifact_path=artifact_file_path,
            model_store=artifact_store
        )

        self.assertIsNotNone(submit_response, "Artifact submission failed.")
        self.assertTrue(url_exists(self.mc.output_data), f"Artifact URL does not exist: {self.mc.output_data}")


if __name__ == "__main__":
    unittest.main()
