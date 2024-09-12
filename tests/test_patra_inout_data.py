from patra_model_card.patra_model_card import *
import json
import unittest
from jsonschema import validate
import os.path

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                           'patra_model_card/schema/schema.json')


class ModelCardTestCase2(unittest.TestCase):

    def setUp(self) -> None:

        self.model_cards = [
            ModelCard(
                name="icicle-camera-traps",
                version="0.1",
                short_description="Camera Traps CNN inference model card",
                full_description="Camera Traps CNN full descr inference model card",
                keywords="cnn, pytorch, icicle",
                author="Joe",
                input_data="https://archive.ics.uci.edu/dataset/2/adult",
                input_type="image",
                output_data="https://archive.ics.uci.edu/dataset/2/adult",
                category="classification",
                ai_model=AIModel(
                    name="DenseNet",
                    version="v0.1",
                    description="DenseNet CNN model",
                    owner="PyTorch",
                    location="pytorch.org",
                    license="testLicence",
                    framework="tensorflow",
                    model_type="cnn",
                    test_accuracy=0.89
                ),
                bias_analysis=BiasAnalysis(0.1, 0.2),
                xai_analysis=ExplainabilityAnalysis("XAI Test 2", [Metric("xai1", 0.5), Metric("xai2", 0.8)])
            ),
            ModelCard(
                name="resnet-classification",
                version="0.2",
                short_description="ResNet model card",
                full_description="ResNet model card for image classification",
                keywords="cnn, resnet, pytorch",
                author="Jane",
                input_data="10.5281/zenodo.11179653",
                input_type="image",
                output_data="10.5281/zenodo.11179653",
                category="classification",
                ai_model=AIModel(
                    name="ResNet",
                    version="v0.2",
                    description="ResNet CNN model",
                    owner="PyTorch",
                    location="pytorch.org",
                    license="testLicenceV2",
                    framework="tensorflow",
                    model_type="cnn",
                    test_accuracy=0.92
                ),
                bias_analysis=BiasAnalysis(0.05, 0.15),
                xai_analysis=ExplainabilityAnalysis("XAI Test ResNet", [Metric("xai1", 0.7), Metric("xai3", 0.9)])
            ),
            ModelCard(
                name="resnet-classification",
                version="0.3",
                short_description="ResNet model card",
                full_description="ResNet model card for image classification",
                keywords="cnn, resnet, pytorch",
                author="Jane",
                input_data="",
                input_type="image",
                output_data="",
                category="classification",
                ai_model=AIModel(
                    name="ResNet",
                    version="v0.3",
                    description="ResNet CNN model",
                    owner="PyTorch",
                    location="pytorch.org",
                    license="testLicenceV2",
                    framework="tensorflow",
                    model_type="cnn",
                    test_accuracy=0.92
                ),
                bias_analysis=BiasAnalysis(0.05, 0.15),
                xai_analysis=ExplainabilityAnalysis("XAI Test ResNet", [Metric("xai1", 0.7), Metric("xai3", 0.9)])
            )
        ]
        self.mc_save_locations = ["./test_mc_1.json", "./test_mc_2.json", "./test_mc_3.json"]

    def test_schema_compatibility(self):
        """
        This tests whether the python classes are in line with the provided model card schema.
        """
        with open(SCHEMA_JSON, 'r') as schema_file:
            schema = json.load(schema_file)

        for model_card in self.model_cards:
            with self.subTest(model_card=model_card.name):
                mc_json = json.dumps(model_card, cls=ModelCardJSONEncoder, indent=4)
                try:
                    validate(json.loads(mc_json), schema)
                except:
                    self.assertTrue(False)

    def test_validate(self):
        """
        This tests the validate function for multiple ModelCard objects.
        """
        for model_card in self.model_cards:
            with self.subTest(model_card=model_card.name):
                is_valid = model_card.validate()
                self.assertTrue(is_valid, f"Validation should fail for {model_card.name}")

    if __name__ == '__main__':
        unittest.main()