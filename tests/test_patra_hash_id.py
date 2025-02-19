import hashlib
import os

import unittest
from unittest.mock import patch, MagicMock
import requests

from patra_toolkit import ModelCard, AIModel, BiasAnalysis, ExplainabilityAnalysis, Metric

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                           'patra_toolkit/schema/schema.json')


class ModelCardTestCase2(unittest.TestCase):

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
        self.mc_save_location = "./test_mc.json"

    @patch('requests.get')
    def test_get_hash_id_success(self, mock_get):
        """Test that id is set correctly when server responds successfully."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"id": "icicle-camera-traps_0.1_joe"}

        model_card = ModelCard(
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

        model_card.id = model_card._get_hash_id("http://127.0.0.1:5002")
        self.assertEqual(model_card.id, {"id": "icicle-camera-traps_0.1_joe"})
        print("Success case id:", model_card.id)

    @patch('requests.get')
    def test_get_hash_id_server_down(self, mock_get):
        """Test ID generation when server is down."""
        mock_get.side_effect = requests.exceptions.RequestException("Server is down")

        model_card = ModelCard(
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
        model_card.id = model_card._get_hash_id("http://127.0.0.1:5002")

        self.assertEqual(model_card.id, {"error": "An unexpected error occurred: Server is down"})
        print("Server down case id:", model_card.id)

if __name__ == '__main__':
        unittest.main()