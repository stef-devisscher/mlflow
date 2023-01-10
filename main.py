import mlflow
import os
from model import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple

#
# this dummy project will show how to train a model to fit to a sinusoidal wave with a given amplitude
#

EPSILON = 0.2 # the noise parameter

# define out 'dataset'
x = torch.linspace(-np.pi, np.pi, 1000)
y = torch.sin(x)
y += torch.rand(size=y.shape) * EPSILON
# save an image of our dataset to samples.png
plt.plot(x, y)
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.savefig("samples.png")

# create a train and validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# define the hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 100

# intialize a model
model = ModelB()

# define the loss function
loss_fn = torch.nn.MSELoss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def random_sample(x: torch.Tensor, y: torch.Tensor, size=BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from the dataset.
    
    :param x: The x features for the sinus function
    :param y: The y features for the sinus function
    :param size: How many feature - value samples to create
    :return: (features, targets) where the features are the input features and the targets
             are the targets the model should predict.
    """
    idx = torch.randint(len(x), (BATCH_SIZE, ))
    # create amplitudes in the range [-1.5, 1.5]
    amplitudes = (torch.rand(len(idx)) - 0.5) * 3
    # create input features being the x value and the amplitude
    xs = x[idx].unsqueeze(1)
    xs = torch.cat((xs, amplitudes.unsqueeze(1)), dim=1)
    # create different amplitudes for the y values
    ys = y[idx] * amplitudes
    return xs, ys

# keep track of the best loss
best_loss = float('inf')

with mlflow.start_run():
    # log the hyperparameters
    mlflow.log_params({
        "lr": LEARNING_RATE, 
        "batch size": BATCH_SIZE,
        "loss fn": "MSELoss",
        "optimizer": "Adam" if type(optimizer) == torch.optim.Adam else "SGD",
        "weight init": False,
        "dropout": False
        })

    # start the training loop
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}/{EPOCHS}")
        epoch_loss = 0
        for _ in range(len(x_train) // BATCH_SIZE):
            # fetch random samples and create a batch
            xs, ys = random_sample(x_train, y_train, BATCH_SIZE)
            xs = xs.unsqueeze(1)
            # zero the gradient
            optimizer.zero_grad()
            # do the predictions with the model
            y_pred = model(xs)
            # calculate the loss and do the backward pass
            loss = loss_fn(y_pred.squeeze(), ys)
            loss.backward()
            optimizer.step()
            # keep track of the loss for logging
            epoch_loss += loss.item()

        # average the loss over the epoch
        epoch_loss /= len(x_train) // BATCH_SIZE

        # calculate the validation metrics
        with torch.no_grad():
            xs, ys = random_sample(x_val, y_val, len(x_val))
            y_pred = model(xs)
            val_loss = loss_fn(y_pred.squeeze(), ys)

        # log the train and validation loss
        mlflow.log_metric("Train/Loss", epoch_loss, step=epoch)
        mlflow.log_metric("Val/Loss", val_loss, step=epoch)

        if val_loss < best_loss:
            state_save = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            # save this model as the current best model
            torch.save(state_save, "logs/model.pt")
            best_loss = val_loss
            print("NEW BEST")
            mlflow.log_artifact("logs/model.pt")
