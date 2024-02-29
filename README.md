# PTaRL

Welcome to the PyTorch Lightning implementation of PTaRL: Prototype-based Tabular Representation Learning via Space Calibration ([ICLR'2024](https://openreview.net/forum?id=G32oY4Vnm8&referrer=%5Bthe%20profile%20of%20Hangting%20Ye%5D(%2Fprofile%3Fid%3D~Hangting_Ye1))). 

This project is an unofficial implementation and was developed to mirror the methods described in the PTaRL paper as closely as possible. 

We have observed discrepancies in the loss function between the paper under review and the code available in the official GitHub repository.

Our implementation is based on the latter.

Since this is an unofficial implementation, updates will be made to align with future official releases or the camera-ready version of the PTaRL paper.

# Getting Started

```sh
pip install -r requirements.txt
```

# Usage

PTaRL employs a two-phase learning approach.

The methodology involves an initial phase of ordinal supervised learning, followed by a second phase that incorporates supervised learning with space calibration. 

To navigate through these phases within the PTaRL framework, the methods set_first_phase() and set_second_phase() are provided for seamless transition.

The PTaRL Lightning Module is designed to be flexible, supporting custom datasets and models under the condition that the dataset's \__getitem__ method returns a tuple containing the input data x and its label y, and the model outputs an embedding vector.

### Example

Below is a simplified example demonstrating how to apply PTaRL with a custom dataset and model:


```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ptarl_lightning import PTARLLightning
from pytorch_lightning import Trainer

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=0, stratify=labels)

# Creating custom dataset instances for training and testing
train_ds = CustomDataset(X_train, y_train)
test_ds = CustomDataset(X_test, y_test)

# DataLoader setup
batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# PTaRL Lightning module instantiation
input_dims = data.shape[1]
emb_dims = 198  # Dimension of the embedding vector
out_dims = 2    # Assuming a binary classification task
model_hparams = {...}  # Define model hyperparameters here

ptarl = PTARLLightning(input_dims, emb_dims, out_dims, train_dl, CustomModel, model_hparams)

# PyTorch Lightning Trainer initialization
trainer = Trainer(...)  # Initialize Trainer with desired configurations

# Executing the first phase of learning
trainer.fit(ptarl, train_dl, test_dl)

# Transitioning to the second phase
ptarl.set_second_phase()

# Executing the second phase of learning
trainer.fit(ptarl, train_dl, test_dl)

```
# Contributing

Contributions to this implementation are highly appreciated. 

Whether it's suggesting improvements, reporting bugs, or proposing new features, feel free to open an issue or submit a pull request.
