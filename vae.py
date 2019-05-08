import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class VAE(nn.Module):
    def __init__(self, data_dim=784, z_dim=10, hidden_dim=500):
        """
        VAE basic model.
        Args:
            data_dim (int): dimension of flatten input
            z_dim (int): dimension of manifold
            hidden_dim (int): dimension of hidden layers between input and manifold
        """
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.hidden2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden2log_var = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, data_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.hidden2mu(h1), self.hidden2log_var(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def reparam(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparam(mu, log_var)
        return self.decode(z), mu, log_var


def train(data_loader, model, loss_func, epoch):
    model.train()
    train_loss = 0
    for batch_i, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_func(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_i % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_i * len(data), len(data_loader.dataset),
                       100. * batch_i / len(data_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(data_loader.dataset)))


def test(data_loader, model, loss_func):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_func(recon_batch, data, mu, log_var).item()

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)

    config = {
        'epochs': 50,
        'z_dim': 30,
        'input_dim': 784,
        'hidden_dim': 200,
        'batch_size': 128,
        'lr': 0.001,
    }
    epochs = config['epochs']
    batch_size = config['batch_size']
    input_dim = config['input_dim']
    z_dim = config['z_dim']
    hidden_dim = config['hidden_dim']
    lr = config['lr']

    # Create directory for resulting images
    if not os.path.exists('results/reconstruction'):
        os.makedirs('results/reconstruction')

    model = VAE(input_dim, z_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def loss_func(x_reconstructed, x, mu, log_var):
        """
        Loss function for VAE
        Args:
            x_reconstructed (torch.Tenor): decoder output [batch_size, input_size]
            x (torch.Tensor): input data [batch_size, input_size]
            mu (torch.Tensor): [batch_size, z_dim]
            log_var (torch.Tensor): [batch_size, z_dim]

        Returns (torch.Tensor): tensor of single loss value

        """
        # Reconstruction loss
        bce = F.binary_cross_entropy(x_reconstructed, x.view(-1, input_dim), reduction="sum")
        # KL divergence
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kld

    # Load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        train(train_loader, model, loss_func, epoch)
        test(test_loader, model, loss_func)
        with torch.no_grad():
            sample = torch.randn(20, z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(20, 1, 28, 28), f'results/sample_{epoch}.png', nrow=10)