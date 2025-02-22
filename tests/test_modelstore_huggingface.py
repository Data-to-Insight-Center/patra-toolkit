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
        self.location = f"https://huggingface.co/nkarthikeyan/{self.repo_id}/blob/main/{self.repo_id}.pt"
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

        self.mc.ai_model = AIModel(
            name="Model",
            version="1.0",
            description="Model",
            owner="user",
            location=self.location,
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )

    def test_upload(self):
        submit_response = self.mc.submit(patra_server_url=self.patra_server_url,
                                         model=models.resnet50(pretrained=True),
                                         model_format="pt",
                                         model_store="huggingface")

        self.assertIsNotNone(submit_response, "Model Card submission failed.")
        self.assertTrue(url_exists(self.location), f"URL does not exist: {self.location}")

        try:
            self.api.delete_repo(repo_id=self.repo_id, token=self.hf_token)
        except Exception as e:
            self.fail(f"Failed to delete repository: {e}")

if __name__ == "__main__":
    unittest.main()
