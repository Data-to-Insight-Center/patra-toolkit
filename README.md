<div align="center">
  
# Patra Model Card Toolkit

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://patra-toolkit.readthedocs.io/en/latest/)
[![Build Status](https://github.com/Data-to-Insight-Center/patra-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Data-to-Insight-Center/patra-toolkit/actions)
[![PyPI version](https://badge.fury.io/py/patra-toolkit.svg)](https://pypi.org/project/patra-toolkit/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Example Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-to-Insight-Center/patra-toolkit/blob/main/examples/notebooks/GettingStarted.ipynb)

</div>

The **Patra Toolkit** simplifies creating and documenting AI/ML models through a structured schema, encouraging best practices and enhanced transparency. It captures essential metadata—model purpose, development process, performance metrics, fairness, and explainability analyses—and packages them into **Model Cards** that can be integrated into the [Patra Knowledge Base](https://github.com/Data-to-Insight-Center/patra-kg).

## Features
- **Structured Schema** – Helps provide critical model information, including usage, development, and performance.
- **Semi-Automated Descriptive Fields** – Automated scanners capture fairness, explainability, and environment dependencies:
  - *Fairness Scanner* – Evaluates predictions across different groups.  
  - *Explainability Scanner* – Provides interpretability metrics.  
  - *Model Requirements Scanner* – Records Python packages and versions.
- **Validation and JSON Generation** – Ensures completeness and correctness before generating the Model Card as JSON.
- **Backend Storage Support** – Pluggable model store backends enable uploading and retrieving models/artifacts from:
  - *Hugging Face* – Integrates with Hugging Face Hub for model storage.  
  - *GitHub* – Leverages GitHub repositories to store serialized models.  
- **Integration with Patra Knowledge Base:** The Model Cards created using the Patra Toolkit are designed to be added to the [Patra Knowledge Base](https://github.com/Data-to-Insight-Center/patra-kg), which is a graph database that stores and manages these cards.

The Patra Toolkit plays a crucial role in promoting transparency and accountability in AI/ML development by making it easier for developers to create comprehensive and informative Model Cards. By automating certain aspects of the documentation process and providing a structured schema, the Toolkit reduces the barriers to entry for creating high-quality model documentation.

For more information, please refer to the [Patra ModelCards paper](https://ieeexplore.ieee.org/document/10678710).


## Installation

```shell
pip install patra-toolkit
```
For local installation:
```shell
git clone https://github.com/Data-to-Insight-Center/patra-toolkit.git
pip install -e patra-toolkit
```

## Usage

### 1. Create a Model Card
```python
from patra_toolkit import ModelCard

mc = ModelCard(
  name="UCI Adult Data Analysis model using Tensorflow",
  version="0.1",
  short_description="UCI Adult Data analysis using Tensorflow for demonstration of Patra Model Cards.",
  full_description="ML model predicting income for UCI Adult Dataset with fairness & explainability scans.",
  keywords="uci adult, tensorflow, fairness, patra, xai",
  author="Sachith Withana",
  input_type="Tabular",
  category="classification",
  citation="Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20."
)

mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
```

### 2. Initialize an AI/ML Model
```python
from patra_toolkit import AIModel

ai_model = AIModel(
  name="UCI Adult Random Forest model",
  version="0.1",
  description="Census classification with Random Forest",
  owner="swithana",
  location="",
  license="BSD-3 Clause",
  framework="sklearn",
  model_type="random_forest",
  test_accuracy=0.85
)

ai_model.add_metric("Epochs", 100)
ai_model.add_metric("BatchSize", 32)
mc.ai_model = ai_model
```

### 3. Populate Fairness and Explainability
```python
mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)
mc.populate_xai(X_test, columns, clf, n_features=10)
```

### 4. Validate and Save the Model Card
```python
mc.populate_requirements()
mc.validate()
mc.save("uci_adult_card.json")
```

### 5. Upload the Model & Artifact
```python
mc.submit_model(
    patra_server_url=<patra_server_url>,
    model=trained_model,
    file_format="pt",
    model_store="huggingface",
    inference_label="labels.txt"
)

mc.submit_artifact(<artifact_path>)
```

## Examples
Explore the following example notebooks and model cards to learn more about how to use the Patra Model Card Toolkit:
[Notebook Example](./examples/notebooks/GettingStarted.ipynb), [Model Card Example](./examples/model_cards/tesorflow_adult_nn_MC.json)

## License
The Patra Model Card toolkit is developed by Indiana University and distributed under the BSD 3-Clause License. See `LICENSE.txt` for more details.

## Acknowledgements
This research is funded in part through the National Science Foundation under award #2112606, AI Institute for Intelligent CyberInfrastructure with Computational Learning in the Environment (ICICLE), and in part through Data to Insight Center (D2I) at Indiana University.

## Reference
S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024, pp. 1-10, doi: 10.1109/e-Science62913.2024.10678710. Keywords: Analytical models, Vectors, Edge-cloud continuum, Model cards, AI/ML accountability, Provenance
