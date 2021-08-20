<div align="center">
  <img src="images/doc.jpeg">
</div>

Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.

## Training

After downloading the dataset you can train your own model using jupyter notebook or if you would like to use RealNVP with other dataset (like image dataset: None-> I have provided code for downloading and training the RealNVP model, with Cifar-10 dataset.) first go to RealNVP file then use the following bash command:

```
python train.py
```

It will download the cifar-10 dataset and starts the training process.

## Testing

If you want to test the MAF model:


Import neccessary libraries
```python
from torch.utils.data import Dataset
import pandas as pd
import torch

import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import flows as fnn
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
```

Define the basic loader class.

```python
class TTTT_Data(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index])
        return x
    
    def __len__(self):
        return self.len
```

Load the data. In this case I have the following data that contains 77 features.

```python
df = pd.read_parquet("TTTT_DNN_nJ4_nB2_2018.parquet")
data_type = 0 # background
data = df.loc[df[df.columns.values[-1]]==data_type]
data.shape
```

Normalize the data:

```python
scaler = MinMaxScaler()
train_data = scaler.fit_transform(data.values)
train_data = TTTT_Data(train_data)
```


Define Hyperparameters

```python
num_inputs = 8 # Feature size
num_hidden = 64 # Number of hidden layers
num_cond_inputs = False
learning_rate = 1e-3
batch_size = 64
block_size = 7

# Optimizers
gamma = 0.1
step_size = 5 # After 5 epoch decrease learning rate by gamma.

act = 'relu'

```

Define the model

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TTTT_Data(train_data)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)


modules = []

for _ in range(block_size):
    modules += [
        fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
        fnn.BatchNormFlow(num_inputs),
        fnn.Reverse(num_inputs)
    ]

model = fnn.FlowSequential(*modules)
model.to(device)
```

Load weights and Switch to evaluation mode, do the testing: (More info can be found in [main.ipynb](main.ipynb) file.)

```python
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()
```
