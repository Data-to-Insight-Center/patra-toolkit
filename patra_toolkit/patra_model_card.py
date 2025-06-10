import json
import logging
import os.path
import tempfile
from dataclasses import dataclass, field
from json import JSONEncoder
from typing import List, Optional, Dict
from urllib.parse import urlparse, urlunparse

import jsonschema
import pkg_resources
import requests

from .exceptions import PatraIDGenerationError
from .fairlearn_bias import BiasAnalyzer
from .model_store import get_model_store, ensure_package_installed
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
    inference_labels: Optional[str] = ""
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
    citation: Optional[str] = ""
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

    def authenticate(self, username: str, password: str) -> str:
        """
        Authenticates the user using TACC credentials and returns a Tapis access token.

        Args:
            username (str): TACC username.
            password (str): TACC password.

        Returns:
            str: Access token string if authentication is successful.

        Raises:
            Exception: If authentication fails.
        """
        payload = {
            "username": username,
            "password": password,
            "grant_type": "password"
        }

        response = requests.post("https://icicleai.tapis.io/v3/oauth2/tokens",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload))
        response.raise_for_status()
        token_data = response.json()

        jwt_token = token_data["result"]["access_token"]["access_token"]
        print("Authentication successful.")
        print("X-Tapis-Token:", jwt_token)
        return jwt_token

    def save(self, file_location: str) -> None:
        """
        Saves the model card as a JSON file to the specified location.

        Args:
            file_location (str): Path where the model card JSON file will be saved.
        """
        try:
            with open(file_location, 'w', encoding='utf-8') as json_file:
                json_file.write(str(self))
            logging.info(f"Model card created.")
        except IOError as io_err:
            logging.error(f"Failed to save model card: {io_err}")

    def submit(
            self,
            patra_server_url: str,
            token: Optional[str] = None,
            model: Optional[object] = None,
            file_format: Optional[str] = "h5",
            model_store: Optional[str] = "huggingface",
            inference_labels: Optional[str] = None,
            artifacts: Optional[List[str]] = None
    ):
        """
        Submits the model card to the Patra server, optionally uploading the model and artifacts.

        Args:
            patra_server_url (str): The URL of the Patra server.
            token (str): The access token for authentication.
            model (object): The trained model to be uploaded.
            file_format (str): The format in which the model will be saved (default: "h5").
            model_store (str): The model store to use for uploading the model (default: "huggingface").
            inference_labels (str): The inference labels to be uploaded.
            artifacts (List[str]): List of artifacts to be uploaded.

        Returns:
            str: "success" if the submission is successful, None otherwise.

        Example:
            .. code-block:: python

                model_card.submit(
                    patra_server_url="http://localhost:5002",
                    token="access_token",
                    model=model,
                    file_format="h5",
                    model_store="huggingface",
                    inference_labels="inference_label.json",
                    artifacts=["requirements.txt", "README.md"]
                )
        """
        # Validate the model card before submission
        if not self.validate():
            logging.error("ModelCard validation failed.")
            return None

        # Retrieve model ID from the Patra server
        is_uploading_model = (model is not None)
        try:
            self.id = self._get_model_id(patra_server_url, token, is_uploading_model)
            logging.info(f"PID created: {self.id}")
            # Update author in the model card to the authenticated user
            # self.author = self.id.split("-")[0]
        except PatraIDGenerationError as pid_exc:
            logging.error(f"Model submission failed during model ID creation: {pid_exc}")
            return None
        except Exception as e:
            logging.error(f"Model submission failed during model ID creation: {e}")
            return None

        # Upload model, inference labels, and artifacts if requested
        model_upload_location = None
        inference_url = None
        artifact_locations = []
        upload_requested = any([model, inference_labels, artifacts])

        if upload_requested:
            # Retrieve credentials for model upload
            try:
                creds = self._get_credentials(patra_server_url, token, model_store)
                credentials = {"token": creds.get("token"), "username": creds.get("username")}
            except Exception as e:
                logging.error(f"Model submission failed during credential retrieval: {e}")
                return None

            # Serialize and upload the model
            if model is not None:
                try:
                    if file_format.lower() == "h5":
                        ensure_package_installed("tensorflow")
                    elif file_format.lower() in ["pt", "onnx"]:
                        ensure_package_installed("torch")
                    serialized_model = self._serialize_model(model, file_format)
                except Exception as e:
                    logging.error(f"Model submission failed during model serialization: {e}")
                    return None

                # Upload the model to the specified model store
                backend = get_model_store(model_store.lower())
                try:
                    model_upload_location = backend.upload(serialized_model, self.id, credentials)
                    logging.info(f"Model uploaded at: {model_upload_location}")
                    self.ai_model.location = model_upload_location
                    self.output_data = self._extract_repository_link(model_upload_location, model_store)

                    model_card_location = os.path.join(tempfile.gettempdir(), "model_card.json")
                    self.save(model_card_location)
                    model_card_upload_location = backend.upload(model_card_location, self.id, credentials)
                    logging.info(f"Model card uploaded at: {model_card_upload_location}")

                    readme_content = f"# Model Card available at: {model_card_upload_location}"
                    readme_path = os.path.join(tempfile.gettempdir(), "README.md")
                    with open(readme_path, 'w', encoding='utf-8') as readme_file:
                        readme_file.write(readme_content)
                    readme_upload_location = backend.upload(readme_path, self.id, credentials)

                except Exception as e:
                    logging.error(f"Model submission failed during model upload: {e}")
                    try:
                        backend.delete_repo(self.id, credentials)
                        logging.info("Rollback successful: repository deleted.")
                    except Exception as rollback_err:
                        logging.error(f"Rollback failed: {rollback_err}")
                    return None

            # Upload the inference label and artifacts
            if inference_labels is not None:
                try:
                    backend = get_model_store(model_store.lower())
                    if model_store.lower() == "huggingface":
                        ensure_package_installed("huggingface_hub")
                    elif model_store.lower() == "github":
                        ensure_package_installed("PyGithub", "github")
                    inference_url = backend.upload(inference_labels, self.id, credentials)
                    self.ai_model.inference_labels = inference_url
                    logging.info(f"Inference labels uploaded at: {inference_url}")
                except Exception as e:
                    logging.error(f"Model submission failed during inference labels upload: {e}")
                    return None

            # Upload artifacts
            if artifacts is not None:
                try:
                    backend = get_model_store(model_store.lower())
                    for artifact in artifacts:
                        loc = backend.upload(artifact, self.id, credentials)
                        logging.info(f"Artifact '{artifact}' uploaded at: {loc}")
                        artifact_locations.append(loc)
                except Exception as e:
                    logging.error(f"Model submission failed during artifact upload: {e}")
                    return None

        # Submit the model card to the Patra server
        try:
            submission_payload = json.loads(str(self))
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-Tapis-Token"] = token

            response = requests.post(
                f"{patra_server_url}/upload_mc",
                json=submission_payload,
                headers=headers
            )
            response.raise_for_status()
            logging.info("Model Card submitted successfully.")
            return "success"
        except Exception as e:
            logging.error(f"Model submission failed during ModelCard submission: {e}")
            if upload_requested:
                try:
                    backend = get_model_store(model_store.lower())
                    backend.delete_repo(self.id, credentials)
                    logging.info("Rollback successful after ModelCard submission failure.")
                except Exception as rollback_err:
                    logging.error(f"Rollback failed: {rollback_err}. Manual cleanup required.")
            return None

    def _get_model_id(self, patra_server_url: str, token: str, is_uploading_model: bool) -> str:
        """
        Retrieves a new model ID from the Patra server based on author, name, and version.
        If the ID already exists:
          - If a model is being uploaded (is_uploading_model is True), an error is raised.
          - If no model is provided (only artifacts or the card), a warning is logged and the existing ID is returned.
        """
        # Ensure a server URL is provided
        if not patra_server_url:
            raise PatraIDGenerationError("No server URL provided for PID generation.")

        # Attempt to retrieve the model ID from the server
        try:
            headers = {"Content-Type": "application/json"}
            params = {"name": self.name, "version": self.version}
            if token:
                headers["X-Tapis-Token"] = token
            else:
                params["author"] = self.author

            response = requests.get(
                f"{patra_server_url.rstrip('/')}/get_model_id",
                params=params,
                headers=headers,
                proxies={"http": None, "https": None},
                timeout=15
            )
            if response.status_code == 409:
                if is_uploading_model:
                    raise PatraIDGenerationError(
                        "Model ID already exists. Please update the model version."
                    )
                else:
                    logging.warning("Model ID exists, but no model is being uploaded; continuing with existing ID.")
                    existing_data = response.json()
                    return existing_data.get("pid", "unknown-id")
            response.raise_for_status()
            id_data = response.json()
            return id_data["pid"]
        except requests.exceptions.ConnectionError:
            raise PatraIDGenerationError("Patra server is unreachable. Verify if the token is provided and valid.")
        except requests.exceptions.Timeout:
            raise PatraIDGenerationError("Patra server connection timed out.")
        except requests.exceptions.RequestException as req_exc:
            raise PatraIDGenerationError(f"Request failed: {req_exc}")

    def _get_credentials(self, patra_server_url: str, token: str, model_store: str) -> Dict[str, str]:
        endpoint = "/get_huggingface_credentials" if model_store.lower() == "huggingface" else "/get_github_credentials"
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Tapis-Token"] = token

        response = requests.get(
            f"{patra_server_url}{endpoint}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def _serialize_model(self, model, file_format: str) -> str:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"{self.id}.{file_format}")

        # Lazy imports for torch and tensorflow
        torch_module = None
        tf_module = None
        try:
            import torch
            torch_module = torch
        except ImportError:
            torch_module = ensure_package_installed("torch")
        try:
            import tensorflow as tf_mod
            tf_module = tf_mod
        except ImportError:
            tf_module = ensure_package_installed("tensorflow")

        # If using PyTorch, perform a lazy import of torch.nn as torch_nn
        if torch_module is not None:
            try:
                from torch import nn as torch_nn
            except ImportError:
                torch_nn = None
        else:
            torch_nn = None

        # PyTorch model branch
        if torch_module is not None and torch_nn is not None and isinstance(model, torch_nn.Module):
            # Allow saving state_dict to either "pt" or "h5"
            if file_format.lower() in ["pt", "h5"]:
                torch_module.save(model, path)
            elif file_format.lower() == "onnx":
                dummy_input = torch_module.randn(1, 3, 224, 224)
                torch_module.onnx.export(model, dummy_input, path)
            else:
                raise Exception(f"Unsupported format for PyTorch models: '{file_format}'")
        # TensorFlow model branch
        elif tf_module is not None and isinstance(model, tf_module.keras.Model):
            if file_format.lower() == "h5":
                model.save(path, save_format='h5')
            else:
                raise Exception("For TensorFlow models, only 'h5' format is supported.")
        else:
            raise Exception("Unsupported model type or missing required framework.")

        logging.info("Model serialized successfully.")
        return path

    @staticmethod
    def _extract_repository_link(model_upload_location: str, model_store: str) -> str:
        """
        Extracts the repository link from the model upload location.
        """
        parsed = urlparse(model_upload_location)
        if model_store.lower() == "huggingface":
            repo_path = parsed.path.split('/blob/')[0] if '/blob/' in parsed.path else parsed.path
        elif model_store.lower() == "github":
            repo_path = parsed.path.split('/tree/')[0] if '/tree/' in parsed.path else parsed.path
        else:
            repo_path = parsed.path
        return urlunparse((parsed.scheme, parsed.netloc, repo_path, '', '', ''))


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
