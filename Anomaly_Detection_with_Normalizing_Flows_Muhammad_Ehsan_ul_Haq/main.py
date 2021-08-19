from dataset import CustomTrainLoaderLHC
from maf import MAF
import torch


def train(model, dataloader, optimizer, epoch, n_epochs, log_interval=100, device='cpu'):
    """
    Train the model.

    Args:
        model -> Model for training.
        dataloader -> Dataloader.
        optimizer -> optimizer.
        epoch -> Current Epoch number (1 <= epoch < n_epochs)
        n_epochs -> Total Epochs
        log_interval -> Interval after which stats should be displayed.
        device -> Cpu or GPU
    """
    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        x, y = data[0], None
        x = x.view(x.shape[0], -1).to(device)

        loss = - model.log_prob(x, y).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch, n_epochs, i, len(dataloader), loss.item()))


if __name__ == '__main__':
    model = MAF(1, 2100, 16, 1)
    n_epochs = 2
    dl = CustomTrainLoaderLHC('Datasets/events_anomalydetection_tiny_table.h5')
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    for epoch in range(n_epochs):
        train(model, dl, optimizer, epoch+1, n_epochs, 2)

