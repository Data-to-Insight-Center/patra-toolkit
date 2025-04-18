import os
import unittest
from unittest.mock import patch
import requests

from patra_toolkit import ModelCard, AIModel, BiasAnalysis, ExplainabilityAnalysis, Metric

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                           'patra_toolkit/schema/schema.json')


class AuthenticationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.aimodel = AIModel(
            name="DenseNet",
            version="v0.1",
            description="DenseNet CNN model",
            owner="PyTorch",
            location="pytorch.org",
            license="testLicence",
            framework="tensorflow",
            model_type="cnn",
            test_accuracy=0.89
        )
        self.bias_analysis = BiasAnalysis(0.1, 0.2)
        self.xai_analysis = ExplainabilityAnalysis("XAI Test 2", [Metric("xai1", 0.5), Metric("xai2", 0.8)])

        self.mc = ModelCard(
            name="icicle-camera-traps",
            version="0.1",
            short_description="Camera Traps CNN inference model card",
            full_description="Camera Traps CNN full descr inference model card",
            keywords="cnn, pytorch, icicle",
            author="Joe",
            input_data="",
            input_type="image",
            output_data="",
            category="classification",
            ai_model=self.aimodel,
            bias_analysis=self.bias_analysis,
            xai_analysis=self.xai_analysis
        )
        self.mc_save_location = "./test_mc.json"

    @patch("requests.post")
    def test_authenticate_success(self, mock_post):
        mock_response = {
            "result": {
                "access_token": {
                    "access_token": "mocked_token_value"
                }
            }
        }

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        token = self.mc.authenticate("fake_user", "fake_pass")

        self.assertEqual(token, "mocked_token_value")
        mock_post.assert_called_once_with(
            "https://icicleai.tapis.io/v3/oauth2/tokens",
            headers={"Content-Type": "application/json"},
            data='{"username": "fake_user", "password": "fake_pass", "grant_type": "password"}'
        )

    @patch("requests.post")
    def test_authenticate_failure(self, mock_post):
        mock_post.return_value.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")

        with self.assertRaises(requests.exceptions.HTTPError):
            self.mc.authenticate("invalid_user", "invalid_pass")


if __name__ == '__main__':
    unittest.main()
