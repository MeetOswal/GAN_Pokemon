from utils import transformer, batch_size
from model import net_D, net_G
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip', 
                          'c065c0e2593b8b161a2d7873e42418bf6a21106c')
data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)

pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size = batch_size,
    shuffle = True, num_workers = d2l.get_dataloader_workers())

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """ Update Discriminator """
    batch_size = X.shape[0] # 1000
    ones = torch.ones((batch_size, ))
    zeros = torch.zeros((batch_size, ))
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) + 
             loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_G(Z, net_D, net_G, loss, trainer_G):
    """ Update Generator """
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size, ))
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing 'fake_Y' is needed since 'net_D' is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device = "cuda"):
    loss = nn.BCEWithLogitsLoss(reduction = "sum")
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr' : lr, 'betas': [0.5, 0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', 
                            xlim=[1, num_epochs], nrows=2, figsize= (5, 5), 
                           legend = ['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace = 0.3)
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0,1, size = (batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                      d2l.update_G(Z, net_D, net_G, loss, trainer_G), 
                       batch_size)
            Z = torch.normal(0,1, size = (21, latent_dim, 1, 1), device = device)
            fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
            imgs = torch.cat(
                [torch.cat([
                    fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim = 1)
                 for i in range(len(fake_x)//7)], dim = 0)
            animator.axes[1].cla()
            animator.axes[1].imshow(imgs)
            loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
            animator.add(epoch, (loss_D, loss_G))
            print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f},' f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')

def main():
    latent_dim, lr, num_epochs = 100, 0.005, 20
    train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)

if __name__ == "__main__":
    main()




