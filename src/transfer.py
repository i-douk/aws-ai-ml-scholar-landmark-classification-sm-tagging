import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True)
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(
            f"Model {model_name} is not known. List of available models: "
            f"https://pytorch.org/vision/{torchvision_major_minor}/models.html"
        )

    # Get number of features from the original classifier
    num_ftrs = model_transfer.fc.in_features

    # Freeze all parameters
    frozen_parameters = []
    for p in model_transfer.parameters():
        if p.requires_grad:
            p.requires_grad = False
            frozen_parameters.append(p)
    print(f"Froze {len(frozen_parameters)} groups of parameters")

    # Replace the classifier head
    model_transfer.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )

    # Unfreeze the new classifier head
    for p in model_transfer.fc.parameters():
        p.requires_grad = True

    return model_transfer



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
