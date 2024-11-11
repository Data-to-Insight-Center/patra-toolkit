==========================
Patra Model Card Toolkit
==========================

Welcome to the documentation for the **Patra Model Card Toolkit**, a framework designed to simplify the creation, documentation, and management of AI/ML model cards. The toolkit provides features for bias analysis, explainability, and other essential metrics to enhance transparency and accountability in AI/ML models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Overview
========

The Patra Model Card Toolkit is part of the Patra ModelCards framework, which facilitates semi-automated information capture for AI/ML models deployed across edge-cloud environments. It is designed to ensure greater trustworthiness by embedding model cards with fairness and explainability metrics.

Key Features
------------

- **Structured Schema**: Provides a detailed schema to guide users in documenting important aspects of their models.
- **Bias Analysis**: Automatically captures fairness metrics using integrated tools like Fairlearn.
- **Explainability Analysis**: Uses SHAP (SHapley Additive exPlanations) to generate feature importance metrics.
- **Model Lifecycle Management**: Tracks model versions and deployment history.
- **Integration with Patra Knowledge Base**: Supports storing and managing model cards in a graph-based knowledge base.

Installation
============

To install the Patra Model Card Toolkit, follow the instructions in the :doc:`installation` section.

.. code:: console

    $ pip install patra_model_card

Usage
=====

Learn how to create a model card, run bias and explainability analysis, and validate your model card. See the :doc:`usage` section for detailed instructions.


.. toctree::
   :maxdepth: 2

   source/patra_model_card

License
=======

The Patra Model Card Toolkit is developed by Indiana University and is distributed under the BSD 3-Clause License. For more details, see `LICENSE.txt`.

References
==========

S. Withana and B. Plale, "Patra ModelCards: AI/ML Accountability in the Edge-Cloud Continuum," 2024 IEEE 20th International Conference on e-Science (e-Science), Osaka, Japan, 2024.