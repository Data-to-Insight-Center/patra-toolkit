import os
import unittest

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from torchvision import models

from patra_toolkit import ModelCard, AIModel

load_dotenv()


class TestUploadDeleteModelCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Called once before running any tests in this class.
        Sets up a Patra server URL for model-card deletion from Neo4j.
        """
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL")
        cls.api = HfApi()

    def setUp(self):
        """
        Runs before each test method.
        Creates a new ModelCard and a PyTorch model to upload.
        """
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
            owner="Neelesh Karthikeyan",
            location='https://huggingface.co/nkarthikeyan/Test Model/blob/main/Test Model.pt',
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )

        self.mc.ai_model = self.ai_model
        self.resnet_model = models.resnet50(pretrained=True)

    def url_exists(self, url: str) -> bool:
        """
        Checks if a URL exists by issuing an HTTP HEAD request.
        Returns True if the response status code is 200.
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking URL existence: {e}")
            return False

    def test_upload_and_delete_modelcard(self):
        """
        1) Submits the model card, uploading the .pt file to Hugging Face.
        2) Verifies that the generated URL exists.
        3) Deletes the Hugging Face repository.
        """
        # 1) Submit the model card (upload .pt file to HF)
        result = self.mc.submit(
            patra_server_url=self.patra_server_url,
            model=self.resnet_model,
            storage_backend="huggingface",
            model_format="pt",
        )
        print("Submit result:", result)
        self.assertNotIn("error", result, f"Submission failed with error: {result.get('error')}")

        # 2) Verify that the URL returned by model_storage_url exists
        file_url = self.mc.model_storage_url
        self.assertTrue(self.url_exists(file_url), f"Uploaded file URL does not exist: {file_url}")

        # 3) Delete the repo from Hugging Face
        try:
            raw_id = file_url.removeprefix("https://huggingface.co/")
            parsed_repo_id = raw_id.split("/blob/")[0]
            self.api.delete_repo(repo_id=parsed_repo_id, token=os.environ.get("HF_HUB_TOKEN"))
        except Exception as e:
            self.fail(f"Failed to delete repo from Hugging Face: {e}")

        print("Test completed successfully: Model uploaded, URL verified, and repository deleted.")


if __name__ == "__main__":
    unittest.main()
