import json
import logging
import os.path
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict

import jsonschema
import pkg_resources
import requests

from .exceptions import PatraIDGenerationError
from .exceptions import PatraSubmissionError
from .fairlearn_bias import BiasAnalyzer
from .shap_xai import ExplainabilityAnalyser

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'schema.json')
logging.basicConfig(level=logging.INFO)

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
    Represents a documented model card containing metadata, analyses, and requirements
    for an AI model. It includes fields for describing the model, performing bias and
    explainability analyses, and validating schema compliance.

    Args:
        name (str): The name of the model card.
        version (str): The model card's version.
        short_description (str): A brief description of the model card.
        full_description (str): A comprehensive description of the model card.
        keywords (str): Comma-separated keywords for searchability.
        author (str): The model's creator or owner.
        input_type (str): Type of input data (e.g., "Image", "Text").
        category (str): The category of the model (e.g., "Classification", "Regression").
        input_data (Optional[str]): Description of the model's input data.
        output_data (Optional[str]): Description of the model's output data.
        foundational_model (Optional[str]): Reference to any foundational model used.
        ai_model (Optional[object]): Reference to an `AIModel` instance containing model details.
        bias_analysis (Optional[object]): Reference to a `BiasAnalysis` instance containing bias metrics.
        xai_analysis (Optional[object]): Reference to an `ExplainabilityAnalysis` instance with interpretability metrics.
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
    ai_model: Optional[object] = None
    bias_analysis: Optional[object] = None
    xai_analysis: Optional[object] = None
    model_requirements: Optional[List[str]] = None
    id: Optional[str] = field(init=False, default=None)

    def __str__(self) -> str:
        """
        Returns:
            str: A JSON-formatted string representation of the model card.
        """
        return json.dumps(self.__dict__, cls=ModelCardJSONEncoder, indent=4, separators=(',', ': '))

    def populate_bias(self,
                      dataset,
                      true_labels,
                      predicted_labels,
                      sensitive_feature_name,
                      sensitive_feature_data,
                      model) -> None:
        """
        Calculates and stores fairness metrics for the model.

        Args:
            dataset (object): The dataset used for bias analysis.
            true_labels (list): The ground truth labels.
            predicted_labels (list): The model's predictions.
            sensitive_feature_name (str): The name of the sensitive attribute.
            sensitive_feature_data (list): Values for the sensitive feature.
            model (object): The model being analyzed.
        """
        bias_analyzer = BiasAnalyzer(dataset, true_labels, predicted_labels, sensitive_feature_name,
                                     sensitive_feature_data, model)
        self.bias_analysis = bias_analyzer.calculate_bias_metrics()

    def populate_xai(self,
                     train_dataset,
                     column_names,
                     model,
                     n_features: int = 10) -> None:
        """
        Computes and stores feature importance metrics for explainability.

        Args:
            train_dataset (object): Training dataset used in the analysis.
            column_names (list): Names of the features in the dataset.
            model (object): The model being explained.
            n_features (int): Number of features to analyze. Default is 10.
        """
        xai_analyzer = ExplainabilityAnalyser(train_dataset, column_names, model)
        self.xai_analysis = xai_analyzer.calculate_xai_features(n_features)

    def populate_requirements(self) -> None:
        """
        Gathers package requirements for the model card, excluding certain dependencies.
        """
        exclude_packages = {"shap", "fairlearn"}
        installed_packages = pkg_resources.working_set
        packages_list = sorted([f"{pkg.key}=={pkg.version}" for pkg in installed_packages])
        self.model_requirements = [
            pkg for pkg in packages_list
            if pkg.split("==")[0] not in exclude_packages
        ]

    def validate(self) -> bool:
        """
        Validates the model card against a predefined JSON schema.

        Returns:
            bool: True if the model card is valid according to the schema, False otherwise.
        """
        mc_json_str = str(self)
        try:
            with open(SCHEMA_JSON, 'r', encoding='utf-8') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(instance=json.loads(mc_json_str), schema=schema)
            logging.info("Model card validation successful.")
            return True
        except jsonschema.ValidationError as val_err:
            logging.error(f"Model card validation error: {val_err.message}")
            return False
        except Exception as exc:
            logging.error(f"Unexpected error during validation: {exc}")
            return False

    def submit(self, patra_server_url: str) -> dict:
        """
        Validates and submits the model card to the specified Patra server.

        Args:
            patra_server_url (str): The Patra server URL for model card submission.

        Returns:
            dict: The server's JSON response on success, or an error message on failure.
        """
        if not patra_server_url:
            logging.error("No Patra server URL provided.")
            return {"error": "No Patra server URL provided."}

        if not self.validate():
            logging.error("Model card validation failed; submission aborted.")
            return {"error": "Model card validation failed."}

        try:
            self.id = self._generate_unique_id(patra_server_url)
            submit_url = f"{patra_server_url}/upload_mc"
            headers = {'Content-Type': 'application/json'}
            response = requests.post(submit_url, json=json.loads(str(self)), headers=headers)
            response.raise_for_status()
            logging.info("Model card submitted successfully.")
            return response.json()
        except PatraIDGenerationError as pid_err:
            raise PatraSubmissionError(f"Unique ID not generated: {pid_err}")
        except requests.exceptions.RequestException as req_err:
            raise PatraSubmissionError(f"Patra server submission failed: {req_err}")

    def _generate_unique_id(self, patra_server_url: str) -> str:
        """
        Generates a unique identifier for the model card based on its metadata.

        Args:
            patra_server_url (str): The Patra server URL used to generate the ID.

        Returns:
            str: A unique identifier for the model card.

        Raises:
            PatraIDGenerationError: If the server fails to generate an ID or returns an error.
        """
        combined_string = f"{self.name}:{self.version}:{self.author}"
        if not patra_server_url:
            raise PatraIDGenerationError("No server URL provided for ID generation.")

        try:
            hash_url = f"{patra_server_url}/get_hash_id"
            headers = {'Content-Type': 'application/json'}
            params = {"author": self.author, "name": self.name, "version": self.version}
            response = requests.get(hash_url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            msg = f"HTTP {response.status_code} error from server: {response.reason}"
            logging.error(msg)
            raise PatraIDGenerationError(msg)
        except requests.exceptions.ConnectionError:
            raise PatraIDGenerationError("Connection to Patra server failed.")
        except requests.exceptions.Timeout:
            raise PatraIDGenerationError("Request to Patra server timed out.")
        except requests.exceptions.RequestException as req_exc:
            raise PatraIDGenerationError(f"Unexpected error: {req_exc}")

    def save(self, file_location: str) -> None:
        """
        Saves the model card as a JSON file to the specified location.

        Args:
            file_location (str): Path where the model card JSON file will be saved.
        """
        try:
            with open(file_location, 'w', encoding='utf-8') as json_file:
                json_file.write(str(self))
            logging.info(f"Model card saved to {file_location}.")
        except IOError as io_err:
            logging.error(f"Failed to save model card: {io_err}")

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
