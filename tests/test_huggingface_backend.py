import os
import unittest
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from torchvision import models
from patra_toolkit import ModelCard, AIModel

load_dotenv()


class TestHuggingFaceBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL")
        cls.api = HfApi()
        cls.hf_token = os.environ.get("HF_HUB_TOKEN")  # Used for repo deletion

    def setUp(self):
        self.location = "https://huggingface.co/nkarthikeyan/Test_Model/blob/main/Test_Model.pt"
        self.mc = ModelCard(
            name="Test Model",
            version="1.0",
            short_description="Test Model",
            full_description="Test Model",
            keywords="model card",
            author="Neelesh Karthikeyan",
            input_type="Image",
            category="classification",
            foundational_model="None"
        )

        self.ai_model = AIModel(
            name="Test Model",
            version="1.0",
            description="Test Model",
            owner="nkarthikeyan",
            location=self.location,
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )
        self.mc.ai_model = self.ai_model
        self.resnet_model = models.resnet50(pretrained=True)

    def url_exists(self, url: str) -> bool:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def test_upload(self):
        response = self.mc.submit(patra_server_url=self.patra_server_url,
                       model=self.resnet_model,
                       model_format="pt",
                       model_store="huggingface")

        self.assertTrue(self.url_exists(self.location), f"URL does not exist: {self.location}")

        try:
            self.api.delete_repo(repo_id=f"nkarthikeyan/Test_Model", token=self.hf_token)
        except Exception as e:
            self.fail(f"Failed to delete repository: {e}")

if __name__ == "__main__":
    unittest.main()
