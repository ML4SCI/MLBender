import os
import util
import torch
import argparse
import torchvision
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from models.real_nvp.real_nvp import RealNVP
from models.real_nvp.real_nvp_loss import RealNVPLoss
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0

    # Dataloader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                     std=[1, 1, 1])])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Model
    net = RealNVP(num_scales=2, in_channels=3, mid_channels=32, num_blocks=4)
    net = net.to(device)


    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    train_loss=[]
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm, train_loss)
        sample(net, 8, device, epoch)

def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, train_loss):
    print('\nEpoch: %d' % epoch)
    net.train()
    for x, _ in trainloader:
        x = x.to(device)
        optimizer.zero_grad()
        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)
        loss.backward()
        util.clip_grad_norm(optimizer, max_grad_norm)
        optimizer.step()
        train_loss.append(loss.item())
        print("Training...")
        
def sample(net, batch_size, device, epoch):

    print('Saving...')
    state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
    os.makedirs('ckpts', exist_ok=True)
    torch.save(state, 'ckpts/'+str(epoch)+'_best.pth.tar')

    z = torch.randn((batch_size, 3, 24, 24), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    # Save samples and data
    images_concat = torchvision.utils.make_grid(x, nrow=int(4 ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_samples', default=8, type=int, help='Number of samples at test time')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--max_grad_norm', type=float, default=100.)

    main(parser.parse_args())
