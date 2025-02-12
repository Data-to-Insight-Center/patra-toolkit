import os
import unittest

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from torchvision import models

from patra_toolkit import ModelCard, AIModel

load_dotenv()


class TestUploadDeleteModelCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Called once before running any tests in this class.
        Ensures HF_HUB_USERNAME, HF_HUB_TOKEN are set for Hugging Face,
        and sets up a Patra server URL for model-card deletion from Neo4j.
        """
        cls.hf_username = os.environ.get("HF_HUB_USERNAME")
        cls.hf_token = os.environ.get("HF_HUB_TOKEN")
        cls.patra_server_url = os.environ.get("PATRA_SERVER_URL")

        if not cls.hf_username or not cls.hf_token:
            raise unittest.SkipTest("Hugging Face credentials not set in environment variables.")

        cls.repo_id = f"{cls.hf_username}/ResNet50_Image_Classification_TestRepo"
        cls.api = HfApi()

    def setUp(self):
        """
        Runs before each test method.
        Creates a new ModelCard and a PyTorch model to upload.
        """
        self.mc = ModelCard(
            name="ResNet50_TestModel",
            version="1.0",
            short_description="PyTorch ResNet50 test upload",
            full_description="Test uploading a ResNet50 model to Hugging Face via Patra Model Card submit()",
            keywords="resnet50, test, huggingface",
            author=self.hf_username,
            input_type="image",
            category="classification"
        )

        self.ai_model = AIModel(
            name="ResNet50_TestModel",
            version="1.0",
            description="ResNet50 test model for integration testing",
            owner=self.hf_username,
            location=f"https://huggingface.co/{self.repo_id}",
            license="Apache-2.0",
            framework="pytorch",
            model_type="cnn",
            test_accuracy=0.76
        )

        self.mc.ai_model = self.ai_model
        self.resnet_model = models.resnet50(pretrained=True)

    def test_upload_and_delete_modelcard(self):
        """
        1) Submits the model card, uploading the .pt file to HF
        2) Verifies the .pt file was uploaded
        3) Deletes the Hugging Face repo
        """
        result = self.mc.submit(
            patra_server_url=self.patra_server_url,
            model=self.resnet_model,
            model_format="pt",
            storage_backend="huggingface"
        )
        print("Submit result:", result)
        self.assertNotIn("error", result, f"Submission failed with error: {result.get('error')}")

        # 2) Verify the .pt file was uploaded
        pt_filename = f"{self.mc.name.replace(' ', '_')}.pt"  # e.g., "ResNet50_TestModel.pt"

        try:
            raw_id = self.mc.model_storage_url.removeprefix("https://huggingface.co/")
            parsed_repo_id = raw_id.split("/blob/")[0]

            downloaded_model_path = hf_hub_download(
                repo_id=parsed_repo_id,
                filename=pt_filename,
                token=self.hf_token
            )
        except Exception as e:
            self.fail(f"Failed to download .pt file from Hugging Face: {e}")

        self.assertTrue(os.path.exists(downloaded_model_path), "Downloaded model file not found.")

        try:
            state_dict = torch.load(downloaded_model_path, map_location=torch.device("cpu"))
            self.assertIsInstance(state_dict, dict, "Loaded .pt is not a valid state dict.")
        except Exception as e:
            self.fail(f"Failed to load .pt file in PyTorch: {e}")

        # 3) Delete the repo from Hugging Face
        try:
            raw_id = self.mc.model_storage_url.removeprefix("https://huggingface.co/")
            parsed_repo_id = raw_id.split("/blob/")[0]
            self.api.delete_repo(repo_id=parsed_repo_id, token=self.hf_token)
        except Exception as e:
            self.fail(f"Failed to delete repo from Hugging Face: {e}")

        print("Test completed successfully: Model uploaded, verified, and deleted from Hugging Face.")


if __name__ == "__main__":
    unittest.main()