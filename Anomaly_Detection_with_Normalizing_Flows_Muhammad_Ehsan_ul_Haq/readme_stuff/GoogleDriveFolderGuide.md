# Google Drive Folder Guide

This project has all the notebooks, model weights, dataset files etc. stored in a [google drive folder](https://drive.google.com/drive/u/1/folders/1WKKG1v3bnAGqs82a4B1lOrI6Y1S0YAXO).
This guide explains the purpose of all the importants files.


## Table of Contents

* [Notebooks](#notebooks)
* [Dataset Files](#dataset-files)
* [Model Weights](#model-weights)
* [Stats Files](#stats-files)

## Notebooks
These are colab notebooks that make use of this repository. Reason for not adding these notebooks to the repo was that there are various saved versions of each notebook that make use of colab's checkpointing feature. If these notebooks are accessed via drive all those versions will also be accessible.

* LHC_MAF_Training.ipynb: This notebook is used for training the model (you can use both datasets, signal or background).
* LHC_MAF_Inference.ipynb: This notebook is used to perform anomaly detection.
* LHC_Generation.ipynb: This notebook is used to sample new data from the modelled distribution (Both Reconstruction and Generation are carried out here).
* LHC_Olympics_CustomLoader_Test.ipynb: This notebook compares various methods to load the dataset and notes the time for each method. (Custom DataLoader that I made for this project performs the best in terms of loading time).

## Dataset Files
The dataset by default is in 'fixed form' (one of different ways a hdf5 file can be saved). The problem with fixed form is that it doesn't allow random access, for this purpose each file was converted into 'table' format (another way of saving an hdf5 file). Because table format introduces indices, this causes the dataset files to become huge (around 17 gb). In order to reduce the size, and to make it easy to benchmark your results on the dataset few modifications were made.

* Float64 type is converted to Float32.
* Two separate datasets for signal and background are created.
* Each signal or background part is then later divided into train, validation and test datasets.
* Naming convention for these is roughly like: events_anomalydetection_{'background'|'signal'}_table_{'train'|'val'|'test'}.h5 e.g. events_anomalydetection_background_table_train.h5.

## Model Weights
Different Experiments were carried out, so for each experiment the trained weights were stored for reproduciblity of the results.

The saved models can be classified into two categories, one is using dequantization and the other using standard normalization.

Dequantized models can be classified into three more categories:
* Dequantized by using batch statistics (feature wise max min).
* Dequantized by using batch statistics (batch wise max min, meaning that a single max value used for normalizing every feature).
* Dequantized by using persistent statistics (Estimate max and min using some technique, and use it for each and every batch in the training, testing and validation set, also feature wise).
* These models are indicated by using the words: dequantized, dequantized_bwm, dequantized_fwmaxmin respectively.

Standard Normalization models are of two categories:
* First the hyperparameters used are same as the Anode Paper. These ones just have the '_neg_loss' word appended to them.
* Then a bit hyperparamters tuning was done to improve the model. These ones have the 'neg_loss_new' appended to them.

## Stats Files
In order to use persistent max and min values for every batch, the statistics tensors were saved to the drive. These are divided into separate files for min and max depending on which part they belong to, signal or background.