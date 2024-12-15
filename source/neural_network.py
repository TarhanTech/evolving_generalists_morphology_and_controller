"""Module containing the defined Neural Network."""

import uuid
from torch import Tensor
import numpy as np
from torch import nn


class NeuralNetwork(nn.Module):
    """Neural Network"""

    def __init__(self, uid):
        super(NeuralNetwork, self).__init__()
        self.uid: uuid.UUID = uid

        self.bias_size: int = 1
        self.input_size: int = 27
        self.hidden_size: int = 20
        self.output_size: int = 8
        self.total_weigths: int = ((self.input_size + self.bias_size) * self.hidden_size) + (
            (self.hidden_size + self.bias_size) * self.output_size
        )

        self.fc1: nn.Linear = nn.Linear(self.input_size, self.hidden_size).double()
        self.fc2: nn.Linear = nn.Linear(self.hidden_size, self.output_size).double()
        self.tanh: nn.Tanh = nn.Tanh()

    def forward(self, x: Tensor) -> np.ndarray:
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))

        if x.is_cuda: 
            return x.to("cpu").detach().numpy()
        else: 
            return x.detach().numpy()


    def set_nn_params(self, nn_params: Tensor):
        """Method that sets the weights of the neural network."""
        assert (
            nn_params.size(0) == self.total_weigths
        ), f"Expected {self.total_weigths} parameters, but got {nn_params.size(0)}."

        input_weight_size = self.input_size * self.hidden_size
        input_bias_size = self.hidden_size

        hidden_weight_size = self.hidden_size * self.output_size

        self.fc1.weight = nn.Parameter(
            nn_params[:input_weight_size].view(self.hidden_size, self.input_size).clone()
        )
        self.fc1.bias = nn.Parameter(
            nn_params[input_weight_size : input_weight_size + input_bias_size].clone()
        )

        self.fc2.weight = nn.Parameter(
            nn_params[
                input_weight_size
                + input_bias_size : input_weight_size
                + input_bias_size
                + hidden_weight_size
            ]
            .view(self.output_size, self.hidden_size)
            .clone()
        )
        self.fc2.bias = nn.Parameter(
            nn_params[input_weight_size + input_bias_size + hidden_weight_size :].clone()
        )
