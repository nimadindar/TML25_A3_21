import torch.nn as nn
import torchvision.models as models

class ModelZoo:
    """
    Class for loading the model structure with random weights. The allowed models are resnet18, resnet34 and 
    resnet50. The accepted input shape is 32x32x3 and the output dimensionality is specified with num_classes
    argument. To load the model, load_model method should be used.
    Args:
        model_name(string)
        num_classes(int)
    returns:
        model
    """
    def __init__(self, model_name:str, num_classes:int):
        self.model_name = model_name
        self.num_classes = num_classes

        self.allowed_models = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        if not model_name in self.allowed_models.keys():
            raise ValueError("The specified model name is not supported for this experiment.\
                            Please choose one of the following options: 1. resnet18 2. resnet34 3. resnet50")
        
    def load_model(self):
        model = self.allowed_models[self.model_name](weights=None)
        # model = self._modify_model(model)
        model.fc = nn.Linear(model.fc.weight.shape[1], self.num_classes)

        return model

    def _modify_model(self, model):
        """
        Method for modifying the model to accept input shape of 32x32x3 and output vector with desired 
        dimensionality of num_classes.
        Args:
            model
        return:
            modified model
        """
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        model.maxpool = nn.Identity()

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)

        return model