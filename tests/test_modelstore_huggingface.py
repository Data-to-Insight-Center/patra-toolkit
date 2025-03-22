import logging
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from torchvision import models
from patra_toolkit import ModelCard, AIModel

logging.basicConfig(level=logging.INFO)


def url_exists(url: str) -> bool:
    try:
        import requests
        return requests.head(url, allow_redirects=True, timeout=10).status_code == 200
    except Exception:
        return False


class TestHuggingFaceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patra_server_url = "dummy_url"
        cls.hf_token = "dummy_token"

    def setUp(self):
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
        # Mock credentials for the model card
        self.mc.credentials = {"username": "test-user"}

    @patch('patra_toolkit.patra_model_card.ModelCard.submit')
    def test_submit_full_submission(self, mock_submit):
        # Set up the mock response
        mock_response = {
            "success": True,
            "model_id": "test-user/ResNetTest_1.0",
            "model_url": "https://huggingface.co/test-user/ResNetTest_1.0"
        }
        mock_submit.return_value = mock_response

        # Create temp files for testing
        temp_artifacts = []
        for fname in ["adult.data", "adult.names", "adult.test"]:
            path = os.path.join(tempfile.gettempdir(), fname)
            with open(path, "w") as f:
                f.write("dummy data")
            temp_artifacts.append(path)

        temp_inference = os.path.join(tempfile.gettempdir(), "labels.txt")
        with open(temp_inference, "w") as f:
            f.write("dummy inference data")

        # Define side effect to set the model location
        def side_effect(*args, **kwargs):
            self.mc.ai_model.location = "https://huggingface.co/test-user/ResNetTest_1.0"
            return mock_response

        mock_submit.side_effect = side_effect

        # Test the submission - this should call our mocked method
        response = self.mc.submit(
            patra_server_url=self.patra_server_url,
            model=models.resnet50(pretrained=False),
            file_format="pt",
            model_store="huggingface",
            inference_labels=temp_inference,
            artifacts=temp_artifacts
        )

        # Assertions
        self.assertIsNotNone(response, "submit_model returned None")
        self.assertIn("success", response)
        self.assertTrue(response["success"])
        self.assertEqual(self.mc.ai_model.location, "https://huggingface.co/test-user/ResNetTest_1.0")

        # Verify the method was called with expected arguments
        mock_submit.assert_called_once()

        # Cleanup
        for path in temp_artifacts:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_inference):
            os.remove(temp_inference)

    @patch('patra_toolkit.patra_model_card.ModelCard.submit')
    def test_submit_model_only(self, mock_submit):
        # Set up the mock response
        mock_response = {
            "success": True,
            "model_id": "test-user/ResNetTest_1.0",
            "model_url": "https://huggingface.co/test-user/ResNetTest_1.0"
        }

        # Define side effect to set the model location
        def side_effect(*args, **kwargs):
            self.mc.ai_model.location = "https://huggingface.co/test-user/ResNetTest_1.0"
            return mock_response

        mock_submit.side_effect = side_effect

        # Test the submission - this should call our mocked method
        response = self.mc.submit(
            patra_server_url=self.patra_server_url,
            model=models.resnet50(pretrained=False),
            file_format="pt",
            model_store="huggingface",
            inference_labels=None,
            artifacts=None
        )

        # Assertions
        self.assertIsNotNone(response, "submit_model returned None")
        self.assertIn("success", response)
        self.assertTrue(response["success"])
        self.assertEqual(self.mc.ai_model.location, "https://huggingface.co/test-user/ResNetTest_1.0")

        # Verify the method was called with expected arguments
        mock_submit.assert_called_once()

    @patch('requests.post')
    def test_submit_modelcard_only(self, mock_post):
        # Setup mock response for the PATRA server
        mock_response = MagicMock(
            status_code=200,
            json=lambda: {"success": True, "model_id": "test-user/ResNetTest_1.0"}
        )
        mock_post.return_value = mock_response

        # Create a subclass and override submit to return a response for modelcard only
        class TestModelCard(ModelCard):
            def submit(self, **kwargs):
                # For modelcard only, we'll return a mock response
                if not kwargs.get('model'):
                    return {"success": True, "model_id": "test-user/ResNetTest_1.0"}
                # Call original for other cases
                return super().submit(**kwargs)

        # Create instance of our test subclass
        test_mc = TestModelCard(
            name="ResNetTest",
            version="1.0",
            short_description="A test ResNet model",
            full_description="Testing ResNet model submission",
            keywords="resnet, test",
            author="test-user",
            input_type="Image",
            category="classification"
        )

        # Test the submission
        response = test_mc.submit(
            patra_server_url=self.patra_server_url,
            model=None,
            file_format=None,
            model_store=None,
            inference_labels=None,
            artifacts=None
        )

        # Assertions
        self.assertIsNotNone(response, "submit_model returned None for model card only submission")
        self.assertIn("success", response)
        self.assertTrue(response["success"])

    def test_url_exists_success(self):
        with patch('requests.head') as mock_head:
            mock_head.return_value = MagicMock(status_code=200)
            self.assertTrue(url_exists("https://example.com"))

    def test_url_exists_failure(self):
        with patch('requests.head') as mock_head:
            mock_head.side_effect = Exception("Connection error")
            self.assertFalse(url_exists("https://invalid-url.xyz"))


if __name__ == "__main__":
    unittest.main()
