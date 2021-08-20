# DOCUMENTATION

## Data Loading API

### dataset.CustomTrainLoaderLHC
```python
from dataset import CustomTrainLoaderLHC
dl = CustomTrainLoaderLHC('Datasets/events_anomalydetection_tiny_table.h5', shuffle=False)

for data, label in dl:
    # DO SOMETHING
    pass
```

## Model Creation API

### maf.MAF
```python
from maf import MAF
'''
n_blocks: number of blocks
input_size: number of input features
hidden_size: number of neurons in hidden layer
batch_norm (bool): perform batch norm between blocks or not 
'''
model = MAF(n_blocks, input_size, hidden_size, n_hidden, batch_norm=False)
# All three parts (Training, Reconstruction and Sampling) assume data is normalized or dequantized.
# Training
loss = -model.log_prob(data).mean(0)

optim.zero_grad()
loss.backward()
optim.step()

# Reconstruction
latent_rep = model(data)
data_recon = model.inverse(latent_rep)

# Sampling
'''
n_samples: number of samples to generate
'''
latent_rep_sample = model.base_dist.sample((n_samples,))
sampled_data = model.inverse(latent_rep_sample)
```