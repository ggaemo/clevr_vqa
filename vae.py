import argparse
import torch

from torch import nn, optim
from torch.nn import functional as F

from torchvision.utils import save_image
import torchvision.utils

from tensorboardX import SummaryWriter

import dataloader


parser = argparse.ArgumentParser()
parser.add_argument('-datasetname', type=str)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-epochs', type=int, default=500)
parser.add_argument('-device_num', type=int, default=0)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('-z_dim', type=int)
parser.add_argument('-enc_layer', type=int, nargs='+')
parser.add_argument('-dec_layer', type=int, nargs='+')
parser.add_argument('-beta', type=float, default=1.0)
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device_num))


train_loader, test_loader, input_dim, num_channel = dataloader.load_data(args.datasetname,
                                                          args.batch_size)

class VAE(torch.nn.Module):
    def __init__(self, input_dim, encoder_layer_config, z_dim, decoder_layer_config):
        super(VAE, self).__init__()

        self.encoder_layer = list()
        self.decoder_layer = list()
        prev_dim = input_dim

        for dim in encoder_layer_config:
            self.encoder_layer.append(nn.Linear(prev_dim, dim))
            self.encoder_layer.append(nn.ReLU())
            prev_dim = dim

        self.encoder_output = nn.Sequential(*self.encoder_layer)

        self.z_mu = nn.Linear(dim, z_dim)
        self.z_logvar = nn.Linear(dim, z_dim)

        if decoder_layer_config is None:
            decoder_layer_config = reversed(encoder_layer_config)

        prev_dim = z_dim
        for dim in decoder_layer_config:
            self.decoder_layer.append(nn.Linear(prev_dim, dim))
            self.decoder_layer.append(nn.ReLU())
            prev_dim = dim

        self.decoder_output = nn.Sequential(*self.decoder_layer)

        self.recon_logit = nn.Linear(dim, input_dim)

    def encode(self, x):
        out = self.encoder_output(x)

        mu = self.z_mu(out)
        logvar = self.z_logvar(out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        out = self.decoder_output(z)
        recon = torch.sigmoid(self.recon_logit(out))
        return recon

    def forward(self, x):
        x = x['image']
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        recon = self.decode(z)
        return recon, mu, log_sigma

model = VAE(input_dim, args.enc_layer, args.z_dim, args.dec_layer).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    RECON = torch.sum(torch.abs(recon_x - x))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ELBO = RECON + KLD * args.beta
    return ELBO, RECON, KLD



def train(epoch):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(-1, input_dim ** 2 * num_channel)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)), end="\r", flush=True)

    train_loss /= len(train_loader.dataset)
    train_recon_loss /= len(train_loader.dataset)
    train_kl_loss /= len(train_loader.dataset)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    return train_loss, train_recon_loss, train_kl_loss


def test(epoch):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = data.view(-1, input_dim ** 2 * num_channel)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu,
                                                       logvar)
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(args.batch_size, num_channel, input_dim,
                                                  input_dim)[:n],
                                      recon_batch.view(args.batch_size, num_channel, input_dim,
                                                       input_dim)[:n]])
                # dummy_img = torch.rand(32, 3, 64, 64)
                # comparison = torchvision.utils.make_grid(dummy_img, normalize=True,
                #                                  scale_each=True)
                # comparison = torchvision.utils.make_grid(comparison, nrow=2)
                writer.add_image('recon', comparison, epoch)

                # save_image(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, test_recon_loss, test_kl_loss


if __name__ == "__main__":
    log_dir = '{}_{}_{}_{}'.format(args.datasetname,
                                '-'.join([str(x) for x in args.enc_layer]),
                                args.z_dim,
                                args.beta)

    writer = SummaryWriter(log_dir)

    print(model)

    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon_loss, train_kl_loss = train(epoch)
        test_loss, test_recon_loss, test_kl_loss = test(epoch)
        writer.add_scalars('data/loss', {'train': train_loss,
                                         'test': test_loss}, epoch)
        writer.add_scalars('data/recon_loss', {'train': train_recon_loss,
                                         'test': test_recon_loss}, epoch)
        writer.add_scalars('data/kl_loss', {'train': train_kl_loss,
                                               'test': test_kl_loss}, epoch)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample)
            writer.add_image('sample', sample.view(64, 1, 28, 28), epoch)

        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     mu, logvar = model.encode(sample)
        #
        #     logvar_mean = torch.mean(logvar, dim=0)
        #     z_dim_significant = torch.argsort(logvar_mean, descending=True)
        #
        #     significant_dims = 20
        #     mu_1 = mu[0]
        #     logvar_1 = logvar[0]
        #     z_val_range = torch.linspace(-3.0, 3.0, 5)
        #     mu_1 = mu_1.expand(len(z_val_range) * significant_dims)
        #     for z_dim in z_dim_significant[:significant_dims]:
        #
        #
        #
        #     writer.add_image('sample', sample.view(64, 1, 28, 28), epoch)