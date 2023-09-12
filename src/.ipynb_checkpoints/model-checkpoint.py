import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        # super().__init__()
        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        # Input: 3x224x224
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,3), stride = (1,1), padding=(2,1)),
            # nn.LeakyReLU(negative_slope=0.2),
            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 24x112x112
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 62x56x56
                
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64x28x28
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 128x14x14
            
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256x7x7
            
            
            nn.Flatten(),
            
            nn.Linear(80*7*7, 500), 
            # nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(500, num_classes),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
