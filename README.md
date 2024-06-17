# ICICLE Model Card library

### Installation
In your colab notebook run the following command to install the Model Card
```shell
!pip install 'git+https://github.com/swsachith/icicle_model_card.git'
```

For developer run the following instead to install locally:
```shell
pip install -e <local_git_dir>/icicle_model_card
```

### How to use the library
Examples are available in the examples/notebook directory. 

##### Adding the model metadata

1. Import the Model Card
```python
from icicle_model_card.icicle_model_card import ModelCard, AIModel
```

2. Initialize the Model Card
```python
mc = ModelCard(
            name="UCI Adult Data Analysis via Random Forest",
            version="0.1",
            short_description="UCI Adult Data analysis using SKLearn and Random Forest",
            full_description="Using a Random Forest to train on UCI Adult Data Analysis",
            keywords="uci adult, sklearn, random_forest, explainability, fairness, fairlearn, shap",
            author="Sachith Withana"
        )
```

3. Add details to the Model Card
```python
mc.input_data = 'https://archive.ics.uci.edu/dataset/2/adult'
mc.output_data = 'https://github.iu.edu/swithana/mcwork/rf_sklearn/adult_model.pkl'
```

4. Add model information
```python
ai_model = AIModel(
            name="UCI Adult Random Forest model",
            version="0.1",
            description="Census classification problem using Random Forest",
            owner="Sachith Withana",
            location="https://github.iu.edu/swithana/mcwork/randomforest/adult_model.pkl",
            license="BSD-3 Clause",
            framework="sklearn",
            model_type="random_forest",
            test_accuracy=accuracy,
            model_structure = None
        )
mc.ai_model = ai_model
```

Types of framework entries provided:
- tensorflow
- sklearn
- pytorch
- other

Types of model types provided:
- cnn
- decision_tree
- dnn
- rnn
- svm
- kmeans
- random_forest
- llm
- lstm
- other

You can also add custom metrics: 
```python
model_metrics = [Metric("Test loss", 0.7)]
ai_model.metrics = model_metrics
```

##### Running fairness scanners

For this, the user needs to provide the sensitive feature as a column along with the following information
- test_dataset
- test_labels
- predictions for the test dataset
- sensitive feature name
- sensitive feature data as a column
- model

The following shows an example usage for fairness information population.
```python
mc.populate_bias(X_test, y_test, predictions, "gender", X_test['sex'], clf)
```

##### Running xai scanners

For this, the user needs to provide the following:
- train_dataset
- column names
- model
- number of top n xai features (default=10)
The following shows an example usage for fairness information population.
```python
mc.populate_xai(X_test, x_columns, model, 10)
```



##### Saving the model card
Verify the information in the model card
```python
print(mc)
```

Save the model card to a given file path. 
```python
from icicle_model_card.icicle_model_card import save_mc
save_mc(mc, <file_path>)
```