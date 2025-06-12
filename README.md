<div align="center">

 <img src="docs/logo.png" alt="Patra Toolkit Logo" width="300"/>

# Patra Model Cards Toolkit

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://patra-toolkit.readthedocs.io/en/latest/)
[![Build Status](https://github.com/Data-to-Insight-Center/patra-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Data-to-Insight-Center/patra-toolkit/actions)
[![PyPI version](https://badge.fury.io/py/patra-toolkit.svg)](https://pypi.org/project/patra-toolkit/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Example Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-to-Insight-Center/patra-toolkit/blob/main/examples/notebooks/GettingStarted.ipynb)

</div>

The Patra Toolkit is a component of the Patra ModelCards framework designed to simplify the process of creating and documenting AI/ML models. It provides a structured schema that guides users in providing essential information about their models, including details about the model's purpose, development process, and performance. The toolkit also includes features for semi-automating the capture of key information, such as fairness and explainability metrics, through integrated analysis tools. By reducing the manual effort involved in creating model cards, the Patra Toolkit encourages researchers and developers to adopt best practices for documenting their models, ultimately contributing to greater transparency and accountability in AI/ML development.

**Tag**: CI4AI, PADI


## Explanation

The Patra Toolkit embeds transparency and governance directly into the training workflow. Integrated scanners collect essential metadata—data sources, fairness metrics, and explainability insights—during model training and then generate a machine‑actionable JSON model card. These cards plug into the Patra Knowledge Base for rich queries on provenance, version history, and auditing. Flexible back‑ends publish models and artifacts to repositories such as Hugging Face or GitHub, automatically recording lineage links to trace every model’s evolution.


## How‑To Guide

### Installation

#### From source

Download the release as source code and unzip it.
```shell
pip install -e <local_git_dir>/patra_toolkit
```

#### Pip

The latest version can be installed from PyPI:

```shell
pip install patra-toolkit
```

## Tutorial

### Building a Patra Model Card

We start with essential metadata like name, version, short description, and so on.

Find the descriptions of the Model Card parameters in the [schema descriptions document](./docs/source/schema_description.md).

```python
from patra_toolkit import ModelCard

mc = ModelCard(
  name="UCI_Adult_Model",
  version="1.0",
  short_description="UCI Adult Data analysis using Tensorflow for demonstration of Patra Model Cards.",
  full_description="We have trained a ML model using the tensorflow framework to predict income for the UCI Adult Dataset. We leverage this data to run the Patra model cards to capture metadata about the model as well as fairness and explainability metrics.",
  keywords="uci adult, tensorflow, explainability, fairness, patra",
  author="neelk",
  input_type="Tabular",
  category="classification",
  foundational_model="None",
   citation="Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI."
)

# Add Model Metadata
mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
mc.output_data = 'https://huggingface.co/patra-iu/neelk-uci_adult_model-1.0'
```

### Initialize an AI/ML Model
Here we describe the model's ownership, license, performance metrics, etc.

```python
from patra_toolkit import AIModel

ai_model = AIModel(
  name="Random Forest",
  version="0.1",
  description="Census classification problem using Random Forest",
  owner="neelk",
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

Patra provides the `demographic_parity_difference` (the difference in the probability of a positive outcome between two groups) and `equalized_odds_difference` (the difference in true positive rates between two groups) using the `fairlearn` library. The explainability metrics are computed using the `shap` library.

```python
# To assess fairness, provide the sensitive feature, test data, labels, and predictions
mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)

# To generate explainability metrics, specify the dataset, column names, model, and number of features
mc.populate_xai(X_test, x_columns, model, top_n=10)
```

The Model Card is validated against the schema to ensure it meets the required structure and content. After validation, you can save the Model Card to a file in JSON format. 

```python
# Capture Python package dependencies and versions
mc.populate_requirements()

# Verify the model card content against the schema
mc.validate()
mc.save(<file_path>)
```

### Submit

The `submit()` method allows you to upload the Model Card, the AI model, and any associated artifacts (like trained models or datasets) to a specified Patra server.

Patra currently supports uploading models (as ".pt" or ".h5" files) and artifacts to Hugging Face and GitHub. Refer the [official documentation](https://patra-toolkit.readthedocs.io/) for more details.

```python
mc.submit(
    patra_server_url=<patra_server_url>,
    model=ai_model,
    file_format="pt",
    model_store="huggingface",
    artifacts=[<artifact1_path>, <artifact2_path>]
)
```

### Persistent Identifier (PID) Generation
Patra assigns each model a PID in the format `<author_id>-<model_name>-<model_version>`. The PID is generated based on the `name`, `version`, and `author` fields of the Model Card. If a name-version conflict arises, increment the `version` field on the Model Card. In case of failure, `submit()` attempts partial rollbacks to avoid orphaned uploads.

For example, the PID for the above model would be `neelk-random_forest-0.1`. This PID can be used to reference the model in the Patra Knowledge Base.

### [Optional] TAPIS Authentication

Patra servers hosted as TAPIS pods require authentication using a JWT (JSON Web Token) for secure access. To generate this token, you must authenticate with your TACC credentials. If you do not already have a TACC account, you can create one at [https://accounts.tacc.utexas.edu/begin](https://accounts.tacc.utexas.edu/begin). Use the Patra `authenticate()` method to obtain an access token for TAPIS-hosted Patra servers:

```python
from patra_toolkit import ModelCard
mc = ModelCard(...)
tapis_token = mc.authenticate(username="<your_tacc_username>", password="<your_tacc_password>")

mc.submit(
    patra_server_url=<tapis_hosted_patra_server_url>,
    token=tapis_token
)
```
The `author` field in the Model Card will automatically be set to your TACC username. This ensures that no two models can have the same author, name, and version combination.

## Examples

Explore the [example notebook](./examples/notebooks/GettingStarted.ipynb) and [example ModelCard](./examples/model_cards/tesorflow_adult_nn_MC.json) to learn more about how to use the Patra Model Card Toolkit.

## License

The **Patra Model Cards Toolkit** is copyrighted by the **Indiana University Board of Trustees** and distributed under the **BSD 3-Clause License**. See the `LICENSE.txt` file for more details.

## Acknowledgements

This research is funded in part through the National Science Foundation under award #2112606, AI Institute for Intelligent CyberInfrastructure with Computational Learning in the Environment (ICICLE), and in part through Data to Insight Center (D2I) at Indiana University.
