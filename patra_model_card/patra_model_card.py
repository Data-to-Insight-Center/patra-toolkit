import hashlib
import json
import os.path
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict

import jsonschema
import pkg_resources
import requests

from .fairlearn_bias import BiasAnalyzer
from .shap_xai import ExplainabilityAnalyser

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'schema.json')


@dataclass
class Metric:
    """
    Data class for storing metric key-value pairs.

    Args:
        key (str): The name of the metric.
        value (str): The value of the metric.
    """
    key: str
    value: str

@dataclass
class AIModel:
    """
    Represents and stores AI model metadata and its performance metrics.

    Args:
        name (str): The name of the model.
        version (str): The version identifier of the model.
        description (str): A detailed description of the model.
        owner (str): The owner of the model.
        location (str): The file path or URL where the model is stored.
        license (str): The license under which the model is distributed.
        framework (str): The framework used to build the model (e.g., TensorFlow, PyTorch).
        model_type (str): The type of model (e.g., classifier, regressor).
        test_accuracy (str): The accuracy of the model on a test dataset.
        model_structure (str): The structure of the model as a dictionary (optional).
        metrics (str): A dictionary storing performance metrics for the model.

    Example:
        .. code-block:: python

            ai_model = AIModel(
                name="Model Name",
                version="1.0",
                description="Model description",
                owner="Model owner",
                location="Model location",
                license="Model license",
                framework="Model framework",
                model_type="Model type",
                test_accuracy=0.95,
                model_structure={},
                metrics={"accuracy": "0.95"}
            )
    """
    name: str
    version: str
    description: str
    owner: str
    location: str
    license: str
    framework: str
    model_type: str
    test_accuracy: float
    model_structure: Optional[object] = field(default_factory=dict)
    metrics: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, key: str, value: str) -> None:
        """
        Adds a performance metric to the model's metrics.

        Args:
            key (str): The name of the metric.
            value (str): The value of the metric.

        Returns:
            None
        """
        self.metrics[key] = value

    def remove_nulls(self, model_structure):
        """
        Recursively removes null values from the model structure.

        Args:
            model_structure (object): The model structure as a dictionary or list.

        Returns:
            object: Model structure with null values removed.
        """
        if isinstance(model_structure, dict):
            return {k: self.remove_nulls(v) for k, v in model_structure.items() if v is not None}
        elif isinstance(model_structure, list):
            return [self.remove_nulls(v) for v in model_structure if v is not None]
        return model_structure

    def populate_model_structure(self, trained_model):
        """
        Populates the `model_structure` attribute from a trained model object.

        Args:
            trained_model (object): A trained machine learning model object.

        Returns:
            None
        """
        if self.framework == 'tensorflow':
            json_structure = json.loads(trained_model.to_json())
            self.model_structure = self.remove_nulls(json_structure)
        else:
            self.model_structure = {}


@dataclass
class BiasAnalysis:
    """
    Class to store results from bias analysis.

    Args:
        demographic_parity_difference (float): The difference in demographic parity between groups.
        equal_odds_difference (float): The difference in equal odds between groups.
    """
    demographic_parity_difference: float
    equal_odds_difference: float


@dataclass
class ExplainabilityAnalysis:
    """
    Class to store explainability metrics.

    Args:
        name (str): Name of the explainability method used.
        metrics (List[Metric]): List of metrics related to explainability analysis.
    """
    name: str
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class ModelCard:
    """
    Represents an AI model card to document model metadata, analyses, and requirements.

    Args:
        name (str): The name of the model.
        version (str): The model's version.
        short_description (str): A brief description of the model.
        full_description (str): A detailed description of the model.
        keywords (str): Comma-separated keywords for searchability.
        author (str): The model's creator or owner.
        input_type (str): Type of input data (e.g., "Image", "Text").
        category (str): The category of the model (e.g., "Classification", "Regression").
        input_data (Optional[str]): Description of the model's input data.
        output_data (Optional[str]): Description of the model's output data.
        foundational_model (Optional[str]): Reference to any foundational model used.
        ai_model (Optional[AIModel]): An instance of `AIModel` containing model details.
        bias_analysis (Optional[BiasAnalysis]): Instance of `BiasAnalysis` containing bias metrics.
        xai_analysis (Optional[ExplainabilityAnalysis]): Instance of `ExplainabilityAnalysis` with interpretability metrics.
        model_requirements (Optional[List[str]]): List of required packages and dependencies.
        id (Optional[str]): Unique identifier for the model card, generated upon submission.

    Example:
        .. code-block:: python

            model_card = ModelCard(
                name="Model Name",
                version="1.0",
                short_description="A brief description",
                full_description="A detailed description of the model's purpose and usage.",
                keywords="classification, AI, image processing",
                author="Author Name",
                input_type="Image",
                category="Classification",
                input_data="Images of size 28x28.",
                output_data="Prediction probabilities for classes.",
                foundational_model="Base Model Reference",
                ai_model=AIModel(
                    name="Model Name",
                    version="1.0",
                    description="Detailed model description",
                    owner="Model owner",
                    location="Storage location",
                    license="MIT",
                    framework="TensorFlow",
                    model_type="Classifier",
                    test_accuracy=0.95,
                    model_structure={},
                    metrics={"accuracy": "0.95"}
                ),
                bias_analysis=BiasAnalysis(
                    demographic_parity_difference=0.05,
                    equal_odds_difference=0.1
                ),
                xai_analysis=ExplainabilityAnalysis(
                    name="SHAP",
                    metrics=[Metric(key="Feature A", value="0.1")]
                ),
                model_requirements=["numpy>=1.19.2", "tensorflow>=2.4.1"]
            )
    """
    name: str
    version: str
    short_description: str
    full_description: str
    keywords: str
    author: str
    input_type: str
    category: str
    input_data: Optional[str] = ""
    output_data: Optional[str] = ""
    foundational_model: Optional[str] = ""
    ai_model: Optional[AIModel] = None
    bias_analysis: Optional[BiasAnalysis] = None
    xai_analysis: Optional[ExplainabilityAnalysis] = None
    model_requirements: Optional[List] = None
    id: Optional[str] = field(init=False, default=None)

    def __str__(self):
        """
        Returns a JSON string representation of the model card.

        Returns:
            str: A JSON-formatted string representing the model card.
        """
        return json.dumps(self.__dict__, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))

    def populate_bias(self, dataset, true_labels, predicted_labels, sensitive_feature_name, sensitive_feature_data, model):
        """
        Calculates and stores fairness metrics.

        Args:
            dataset (object): The dataset used for bias analysis.
            true_labels (list): The ground truth labels.
            predicted_labels (list): Model's predictions.
            sensitive_feature_name (str): The name of the sensitive attribute.
            sensitive_feature_data (list): Values for the sensitive feature.
            model (object): The model being analyzed.

        Returns:
            None
        """
        bias_analyzer = BiasAnalyzer(dataset, true_labels, predicted_labels, sensitive_feature_name,
                                     sensitive_feature_data, model)
        self.bias_analysis = bias_analyzer.calculate_bias_metrics()

    def populate_xai(self, train_dataset, column_names, model, n_features=10):
        """
        Computes and stores feature importance metrics.

        Args:
            train_dataset (object): Training dataset used in the analysis.
            column_names (list): Names of the features.
            model (object): The model being explained.
            n_features (int, optional): Number of features to analyze. Default is 10.

        Returns:
            None
        """
        xai_analyzer = ExplainabilityAnalyser(train_dataset, column_names, model)
        self.xai_analysis = xai_analyzer.calculate_xai_features(n_features)

    def populate_requirements(self):
        """
        Collects package requirements for the model card, excluding specific dependencies.

        Returns:
            None
        """
        exclude_packages = {"shap", "fairlearn"}
        installed_packages = pkg_resources.working_set
        packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
        self.model_requirements = [pkg for pkg in packages_list if pkg.split("==")[0] not in exclude_packages]

    def validate(self):
        """
        Validates the model card against a predefined JSON schema.

        Returns:
            bool: True if the model card is valid according to the schema, False otherwise.
        """
        mc_json = self.__str__()
        try:
            with open(SCHEMA_JSON, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(instance=json.loads(mc_json), schema=schema)
            return True
        except jsonschema.ValidationError as e:
            print(e.message)
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def submit(self, patra_server_url):
        """
        Validates and submits the model card to the specified Patra server.

        Args:
            patra_server_url (str): The Patra server URL where the model card should be submitted.

        Returns:
            dict: The server's response as a JSON object.
        """
        if self.validate():
            try:
                self.id = self._get_hash_id(patra_server_url)
                patra_submit_url = f"{patra_server_url}/upload_mc"
                headers = {'Content-Type': 'application/json'}
                response = requests.post(patra_submit_url, json=json.loads(str(self)), headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print("The Patra Server cannot be reached. Please try again.")
                return None
        return {"An error occurred: valid patra_server_url not provided. Unable to upload."}

    def _get_hash_id(self, patra_server_url):
        """
        Generates a unique identifier for the model card based on its metadata.

        Args:
            patra_server_url (str): The Patra server URL used to generate the ID.

        Returns:
            str: A unique hash identifier for the model card.
        """
        combined_string = f"{self.name}:{self.version}:{self.author}"
        try:
            if patra_server_url:
                patra_hash_url = f"{patra_server_url}/get_hash_id"
                headers = {'Content-Type': 'application/json'}
                response = requests.get(patra_hash_url, params={"combined_string": combined_string}, headers=headers)
                response.raise_for_status()
                return response.json()
            else:
                return hashlib.sha256(combined_string.encode()).hexdigest()
        except requests.exceptions.RequestException as e:
            print("Could not connect to the Patra Server, generating the ID locally")
            return hashlib.sha256(combined_string.encode()).hexdigest()

    def save(self, file_location):
        """
        Saves the model card as a JSON file to the specified location.

        Args:
            file_location (str): The path where the model card JSON file will be saved.

        Returns:
            None
        """
        with open(file_location, 'w') as json_file:
            json_file.write(str(self))


class ModelCardJSONEncoder(JSONEncoder):
    """
    Custom JSON Encoder for ModelCard to handle complex objects.

    Methods:
        default: Serializes non-serializable fields.
    """

    def default(self, obj):
        if isinstance(obj, (ModelCard, Metric, AIModel, ExplainabilityAnalysis, BiasAnalysis)):
            return obj.__dict__
        return super().default(obj)
