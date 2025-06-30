import os
import requests
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torchvision import models

from models.load_model import ModelZoo

load_dotenv()
TOKEN = os.getenv("TOKEN")

MODEL_NAME = "resnet18"
METHOD = "GAIRAT"
NUM_EXPERIMENT = 1

SAVED_PATH = f"./results/{METHOD}_method/bestpoint.pth.tar"

allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
with open(SAVED_PATH, "rb") as f:
    try:
        model = ModelZoo(MODEL_NAME, 10).load_model()
        # model = torch.nn.DataParallel(model)
    except Exception as e:
        raise Exception(
            f"Invalid model class, {e=}, only {allowed_models.keys()} are allowed",
        )
    try:
        print(f"Loading model {MODEL_NAME} with method {METHOD}...")
        assert os.path.isfile(SAVED_PATH)
        print(f"Loading the model from path: {SAVED_PATH} ...")

        checkpoint = torch.load(SAVED_PATH)
        state_dict = checkpoint['state_dict']

        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'num_batches_tracked' not in k}

        model.load_state_dict(clean_state_dict)

        torch.save(clean_state_dict, 
                   f"./results/submissions/experiment_{NUM_EXPERIMENT}_method_{METHOD}_model_{MODEL_NAME}.pt")
        print(model)
        model.eval()
        out = model(torch.randn(1, 3, 32, 32))
        print("Loading the model done successfully!")
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")

    assert out.shape == (1, 10), "Invalid output shape"


# Send the model to the server, replace the string "TOKEN" with the string of token provided to you
response = requests.post("http://34.122.51.94:9090/robustness", 
                         files={"file": open(f"./results/submissions/experiment_{NUM_EXPERIMENT}_method_{METHOD}_model_{MODEL_NAME}.pt", "rb")}, 
                         headers={"token": TOKEN, "model-name": MODEL_NAME})

# Should be 400, the clean accuracy is too low
print(response.json())