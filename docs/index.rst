==========================
Patra Model Card Toolkit
==========================

The Patra Toolkit is a component of the Patra ModelCards framework designed to simplify the process of creating and documenting AI/ML models. It provides a structured schema that guides users in providing essential information about their models, including details about the model's purpose, development process, and performance.

.. toctree::
   :maxdepth: 2

   source/patra_toolkit

Features
========

- **Structured Schema:** The Patra Toolkit offers a structured schema to guide users in providing crucial model information.
- **Semi-Automated Information Capture:** The toolkit supports semi-automatic capture of certain descriptive fields.
- **Validation and JSON Generation:** Once a Model Card is created using the Toolkit, it validates the data against the defined schema.
- **Integration with Patra Knowledge Base:** The Model Cards created using the Patra Toolkit are designed to be added to the Patra Knowledge Base.

Getting Started
===============

**(Optional) Create a virtual environment for using Patra Model Card Toolkit**

We recommend creating a new virtual environment using venv before installing patra-model-card.

Installing Patra Model Card
---------------------------

The latest version can be installed from PyPI:

.. code-block:: console

    pip install patra-toolkit

For local installation, clone the repository and install using:

.. code-block:: console

    pip install -e <local_git_dir>/patra_toolkit

Usage
=====

Create a Model Card
-------------------

Find the descriptions of the Model Card parameters in the schema descriptions document.

.. code-block:: python

    from patra-toolkit import ModelCard

    mc = ModelCard(
        name="UCI Adult Data Analysis model using Tensorflow",
        version="0.1",
        short_description="UCI Adult Data analysis using Tensorflow for demonstration of Patra Model Cards.",
        full_description="We have trained a ML model using the tensorflow framework to predict income for the UCI Adult Dataset.",
        keywords="uci adult, tensorflow, explainability, fairness, patra",
        author="Sachith Withana",
        input_type="Tabular",
        category="classification",
        foundational_model="None"
    )

    # Add Model Metadata
    mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
    mc.output_data = 'https://github.iu.edu/swithana/mcwork/rf_sklearn/adult_model.pkl'

Initialize an AI/ML Model
-------------------------

.. code-block:: python

    from patra-toolkit import AIModel

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

Run Fairness and Explainability Scanners
----------------------------------------

.. code-block:: python

    # To assess fairness, provide sensitive feature, test data, labels, and predictions
    mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)

    # To generate explainability metrics
    mc.populate_xai(X_test, x_columns, model, top_n=10)

Validate and Save the Model Card
--------------------------------

.. code-block:: python

    # Verify model card content against schema
    mc.validate()
    mc.save(<file_path>)