<div align="center">

# Patra Model Card Toolkit

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://patra-toolkit.readthedocs.io/en/latest/)
[![Build Status](https://github.com/Data-to-Insight-Center/patra-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Data-to-Insight-Center/patra-toolkit/actions)
[![PyPI version](https://badge.fury.io/py/patra-toolkit.svg)](https://pypi.org/project/patra-toolkit/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Example Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-to-Insight-Center/patra-toolkit/blob/main/examples/notebooks/GettingStarted.ipynb)

</div>

The Patra Toolkit is a component of the Patra ModelCards framework designed to simplify the process of creating and documenting AI/ML models. It provides a structured schema that guides users in providing essential information about their models, including details about the model's purpose, development process, and performance. The toolkit also includes features for semi-automating the capture of key information, such as fairness and explainability metrics, through integrated analysis tools. By reducing the manual effort involved in creating model cards, the Patra Toolkit encourages researchers and developers to adopt best practices for documenting their models, ultimately contributing to greater transparency and accountability in AI/ML development.

---

## Features

1. **Encourages Accountability**  
   - Incorporate essential model information (metadata, dataset details, fairness, explainability) at training time, ensuring AI models remain transparent from development to deployment.

2. **Semi-Automated Capture**  
   - Automated *Fairness* and *Explainability* scanners compute demographic parity, equal odds, SHAP-based feature importances, etc., for easy integration into Model Cards.

3. **Machine-Actionable Model Cards**  
   - Produce a structured JSON representation for ingestion into the Patra Knowledge Base. Ideal for advanced queries on model selection, provenance, versioning, or auditing.

4. **Flexible Repository Support**  
   - Pluggable backends for storing models/artifacts on **Hugging Face** or **GitHub**, unifying the model publishing workflow.

5. **Versioning & Model Relationship Tracking**  
   - Maintain multiple versions of a model with recognized edges (e.g., `revisionOf`, `alternateOf`) using embedding-based similarity. This ensures clear lineages and easy forward/backward provenance.

## Getting Started

#### Installing Patra Model Card
The latest version can be installed from PyPI:
```shell
pip install patra-toolkit
```

For local installation, clone the repository and install using:
```shell
pip install -e <local_git_dir>/patra_toolkit
```

## Usage
### Create a Model Card
Find the descriptions of the Model Card parameters in the [schema descriptions document](./docs/schema_description.md).

```python
from patra_toolkit import ModelCard

mc = ModelCard(
    name="UCI Adult Data Analysis model using Tensorflow",
    version="0.1",
    short_description="UCI Adult Data analysis using Tensorflow for demonstration of Patra Model Cards.",
    full_description="We have trained a ML model using the tensorflow framework to predict income for the UCI Adult Dataset. We leverage this data to run the Patra model cards to capture metadata about the model as well as fairness and explainability metrics.",
    keywords="uci adult, tensorflow, explainability, fairness, patra",
    input_type="Tabular",
    category="classification",
    foundational_model="None"
)

# Add Model Metadata
mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
mc.output_data = 'https://huggingface.co/Data-to-Insight-Center/UCI-Adult'

# Add User Information
mc.populate_user(
    username="neelk",
    orcid="0000-0002-1234-5678",
    name="Neelesh Karthikeyan",
    institution="Indiana University",
    email="neelk@iu.edu"
)
```

### Initialize an AI/ML Model

```python
from patra_toolkit import AIModel

ai_model = AIModel(
  name="UCI Adult Random Forest model",
  version="0.1",
  description="Census classification problem using Random Forest",
  owner="Sachith Withana",
  location="https://github.iu.edu/swithana/mcwork/randomforest/adult_model.pkl",
  license="BSD-3 Clause",
  framework="sklearn",
  model_type="random_forest",
  test_accuracy=accuracy
)

# Populate Model Structure
ai_model.populate_model_structure(trained_model)
mc.ai_model = ai_model

# Add Custom Metrics
ai_model.add_metric("Test loss", loss)
ai_model.add_metric("Epochs", 100)
ai_model.add_metric("Batch Size", 32)
ai_model.add_metric("Optimizer", "Adam")
ai_model.add_metric("Learning Rate", 0.0001)
ai_model.add_metric("Input Shape", "(26048, 100)")
```

### Run Fairness and Explainability Scanners
```python
# To assess fairness, provide the sensitive feature, test data, labels, and predictions
mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)

# To generate explainability metrics, specify the dataset, column names, model, and number of features
mc.populate_xai(X_test, x_columns, model, top_n=10)
```

### Validate and Save the Model Card
```python
# Capture Python package dependencies and versions
mc.populate_requirements()

# Verify the model card content against the schema
mc.validate()
mc.save(<file_path>)
```

## Submit

Use `mc.submit()` to either upload just a model card, an AI model along with the model card, just the artifacts, or all at once!

```python
mc.submit(
    patra_server_url=<patra_server_url>,
    model=<trained_model>,
    file_format="pt",  # or "h5"
    model_store="huggingface",  # or "github"
    inference_labels="labels.txt",
    artifacts=[<artifact1_path>, <artifact2_path>],
    token=<optional_token>  # optional authentication token
)
```

The `token` parameter is **optional**. If your hosted Patra server requires authentication, provide a valid token.

If a name-version conflict arises, increment `mc.version`. In case of failure, `submit()` attempts partial rollbacks to avoid orphaned uploads.

---

## Authentication with TACC Credentials

To authenticate against a Patra server hosted in TAPIS, use Patra's built-in `authenticate()` method to obtain an access token:

```python
from patra_toolkit import ModelCard

mc = ModelCard(...)

tapis_token = mc.authenticate(username="<your_tacc_username>", password="<your_tacc_password>")
```

This will print and return a valid `X-Tapis-Token` (JWT). You can then pass this token to `mc.submit()`:

```python
mc.submit(
    patra_server_url=<tapis_hosted_patra_server_url>,
    model=<trained_model>,
    token=tapis_token
)
```

---

## Examples
Explore the following example notebooks and model cards to learn more about how to use the Patra Model Card Toolkit:
[Notebook Example](./examples/notebooks/GettingStarted.ipynb), [Model Card Example](./examples/model_cards/tesorflow_adult_nn_MC.json)

## License
The Patra Model Card toolkit is developed by Indiana University and distributed under the BSD 3-Clause License. See `LICENSE.txt` for more details.

## Acknowledgements
This research is funded in part through the National Science Foundation under award #2112606, AI Institute for Intelligent CyberInfrastructure with Computational Learning in the Environment (ICICLE), and in part through Data to Insight Center (D2I) at Indiana University.

## Reference
S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024, pp. 1-10, doi: 10.1109/e-Science62913.2024.10678710. Keywords: Analytical models, Vectors, Edge-cloud continuum, Model cards, AI/ML accountability, Provenance
