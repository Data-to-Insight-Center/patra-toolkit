import os
import unittest

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


    def test_input_data_empty(self):
        """
        This function tests whether the input data is empty.
        """
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_input_data_doi(self):
        """
        This function tests whether the input data is doi.
        """
        self.mc.input_data = "10.5281/zenodo.11179653"
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_input_data_url(self):
        """
        This function tests whether the input data is url.
        """
        self.mc.input_data = "https://archive.ics.uci.edu/dataset/2/adult"
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_input_data_string(self):
        """
        This function tests whether the input data is string.
        """
        self.mc.input_data = "cifar10"
        is_valid = self.mc.validate()
        self.assertFalse(is_valid)

    def test_output_data_empty(self):
        """
        This function tests whether the output data is empty.
        """
        self.mc.input_data = "10.5281/zenodo.11179653"
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_output_data_doi(self):
        """
        This function tests whether the output data is doi.
        """
        self.mc.input_data = "10.5281/zenodo.11179653"
        self.mc.output_data = "10.5281/zenodo.11179653"
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_output_data_url(self):
        """
        This function tests whether the output data is url.
        """
        self.mc.input_data = "10.5281/zenodo.11179653"
        self.mc.output_data = "https://archive.ics.uci.edu/dataset/2/adult"
        is_valid = self.mc.validate()
        self.assertTrue(is_valid)

    def test_output_data_string(self):
        """
        This function tests whether the output data is string.
        """
        self.mc.input_data = "10.5281/zenodo.11179653"
        self.mc.output_data = "hcifar10"
        is_valid = self.mc.validate()
        self.assertFalse(is_valid)

if __name__ == '__main__':
        unittest.main()