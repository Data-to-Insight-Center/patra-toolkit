if __name__ == '__main__':

    import torchvision.models as models
    from patra_toolkit import ModelCard, AIModel

    # Initialize the Model Card
    mc = ModelCard(
        name="ResNet50 Image Classification Model",
        version="1.0",
        short_description="Pre-trained ResNet50 model from torchvision for image classification.",
        full_description=(
            "This ResNet50 model is pre-trained on ImageNet and can be used for a variety "
            "of image classification tasks. It demonstrates the use of Patra Model Cards to "
            "document model metadata, bias, and explainability metrics."
        ),
        keywords="ResNet50, image classification, pytorch, ImageNet, model card",
        author="Neelesh Karthikeyan",
        input_type="Image",
        category="classification",
        foundational_model="None"
    )

    # Define input and output data URLs
    mc.input_data = 'https://huggingface.co/datasets/cifar10'
    mc.output_data = 'https://huggingface.co/nkarthikeyan/ResNet50_Image_Classification_Model/blob/main/ResNet50_Image_Classification_Model.pt'

    # Initialize AI Model details
    ai_model = AIModel(
        name="ResNet50 Image Classification Model",
        version="1.0",
        description=(
            "Pre-trained ResNet50 model from torchvision for image classification. "
            "This model achieves approximately 76% top-1 accuracy on ImageNet."
        ),
        owner="Neelesh Karthikeyan",
        location='https://huggingface.co/nkarthikeyan/ResNet50_Image_Classification_Model/blob/main/ResNet50_Image_Classification_Model.pt',
        license="Apache-2.0",
        framework="pytorch",
        model_type="cnn",
        test_accuracy=0.76
    )

    model = models.resnet50(pretrained=True)

    ai_model.populate_model_structure(model)
    mc.ai_model = ai_model
    mc.populate_requirements()

    # Validate the Model Card
    mc.validate()

    # Submit the Model Card
    mc.submit(patra_server_url="http://10.20.227.55:5002", model=model, storage_backend="huggingface", model_format="pt")