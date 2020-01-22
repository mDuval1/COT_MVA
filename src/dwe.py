import numpy as np
import pandas as pd
import ot
import tqdm
import torch
import torch.nn as nn
import multiprocessing as mp

from .utils import compute_dist, w2dist, partial_dist


class KLloss(nn.Module):

    def __init__(self):
        super(KLloss, self).__init__()

    def forward(self, x, y):
        x = torch.clamp(x, min=1e-6, max=1)
        y = torch.clamp(y, min=1e-6, max=1)
        return torch.sum(x * (torch.log(x) - torch.log(y)), dim=tuple(i for i in range(1, x.dim()))).sum()


class DWE:
    def __init__(self, encoder, decoder, regterm, cuda=False, name=None, from_dir=None, lr=1e-3):
        if from_dir is None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.load(from_dir, name)
        self.regterm = regterm
        self.cuda = cuda
        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr) 
        # self._kl_loss = torch.nn.KLDivLoss(reduction='mean')
        # self.kl_loss = lambda x, y: self._kl_loss(input=torch.log(x), target=y)
        self.kl_loss = KLloss()
        self.mse_loss = nn.MSELoss(reduction='sum')
        if cuda:
            self.encoder.cuda()
            self.decoder.cuda()
        
    
    def loss_batch(self, x1, x2, y):
        if self.cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
        phi1 = self.encoder(x1)
        phi2 = self.encoder(x2)
        psi1 = self.decoder(phi1)
        psi2 = self.decoder(phi2)
        d_hat = ((phi1 - phi2)**2).sum(dim=1)
        encoding_term = self.mse_loss(input=d_hat, target=y)
        acc_term = torch.abs(d_hat - y).sum()
        # print((psi1 > 0).detach().cpu().numpy().mean(),
        #         (x1 > 0).detach().cpu().numpy().mean(),
        #         (psi2 > 0).detach().cpu().numpy().mean(),
        #         (x2 > 0).detach().cpu().numpy().mean())
        recons_term = self.kl_loss(psi1, x1) + self.kl_loss(psi2, x2)
        loss = encoding_term + self.regterm * recons_term
        # print(encoding_term, recons_term)
        return loss, encoding_term, recons_term, acc_term

    def train(self, trainloader, has_y, nepochs=10, valloader=None):
        n_items = len(trainloader.dataset)
        for epoch in tqdm.tqdm_notebook(range(nepochs)):
            train_acc = 0
            train_loss = 0
            loss_w = 0
            loss_data = 0
            self.encoder.train()
            self.decoder.train()
            for x1, x2, y in trainloader:
                if has_y:
                    x1 = x1[0]
                    x2 = x2[0]
                loss, w_term, data_term, w_acc = self.loss_batch(x1, x2, y.float())
                train_loss += loss
                train_acc += w_acc
                loss_w += w_term
                loss_data += data_term
                loss /= trainloader.batch_size
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            print(f'Mean train loss : {train_loss/n_items:.2f} (Wasserstein : {loss_w/n_items:.2f} - ' + 
                  f'data : {loss_data/n_items:.2f})')
            print(f'Mean train accuracy : {train_acc/n_items:.2f}')
            if not valloader is None:
                self.evaluate(valloader, has_y=has_y)

    def evaluate(self, dataloader, has_y=True):
        n_items = len(dataloader.dataset)
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            val_acc = 0
            val_loss = 0
            loss_w = 0
            loss_data = 0
            for x1, x2, y in dataloader:
                if has_y:
                    x1 = x1[0]
                    x2 = x2[0]
                loss, w_term, data_term, w_acc = self.loss_batch(x1, x2, y.float())
                val_loss += loss
                val_acc += w_acc
                loss_w += w_term
                loss_data += data_term
                loss /= dataloader.batch_size
            print(f'Mean train loss : {val_loss/n_items:.2f} (Wasserstein : {loss_w/n_items:.2f} - ' + 
                  f'data : {loss_data/n_items:.2f})')
            print(f'Mean train accuracy : {val_acc/n_items:.2f}')

    def save(self, dir, name):
        torch.save(self.encoder, f'{dir}/encoder_{name}')
        torch.save(self.decoder, f'{dir}/decoder_{name}')

    def load(self, dir, name):
        self.encoder = torch.load(f'{dir}/encoder_{name}')
        self.decoder = torch.load(f'{dir}/decoder_{name}')




class DWE_dataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, N, from_file=None, has_y=True, p=2):
        self.N = N
        self.dataset = dataset
        self.has_y = has_y
        self.p = p
        
        self.width_im = self.dataset[0][0].shape[1]
        xx,yy = np.meshgrid(np.arange(self.width_im), np.arange(self.width_im))
        xy = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
        self.M = ot.dist(xy, xy, metric=partial_dist(p))
        
        if from_file is None:
            self._reset_pairs()
        else:
            self.load(from_file)
        
    def _reset_pairs(self):
        n = len(self.dataset)
        self.pairs = {}
        pbar = tqdm.tqdm_notebook(total=self.N)
        while(len(self.pairs) < self.N):
            i1, i2 = np.random.choice(n, size=2)
            y = compute_dist(self.dataset[i1][0], self.dataset[i2][0], self.M)
            if (i1, i2) not in self.pairs.keys():
                self.pairs[(i1, i2)] = y
                pbar.update(1)
        pbar.close()
        self.id_to_pair = dict(zip(np.arange(self.N), list(self.pairs.keys())))

    def save(self, path):
        df = []
        for i in range(self.N):
            i1, i2 = self.id_to_pair[i]
            y = self.pairs[(i1, i2)]
            df.append([i1, i2, y])
        df = pd.DataFrame(df, columns=['im1', 'im2', 'dist'])
        df.to_csv(path)
    
    def load(self, df):
        self.pairs = {}
        for i in range(self.N):
            i1, i2, y = df.loc[i]
            self.pairs[(int(i1), int(i2))] = y
        self.id_to_pair = dict(zip(np.arange(self.N), list(self.pairs.keys())))
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, i):
        i1, i2 = self.id_to_pair[i]
        y = self.pairs[(i1, i2)]
        x1, x2 = self.dataset[i1], self.dataset[i2]
        if self.has_y:
            x1 = (x1[0] / x1[0].sum(), x1[1])
            x2 = (x2[0] / x2[0].sum(), x2[1])
        else:
            x1 /= x1.sum()
            x2 /= x2.sum()
        return x1, x2, y


class DWE_dsGaussian(DWE_dataset):
    
    def __init__(self, ds, ys, N=1000, from_file=None):
        self.N = N
        self.ds = torch.tensor(ds).float()
        self.ys = ys
        
        if from_file is None:
            self._initPairs()
        else:
            self.load(from_file)
        
    def _initPairs(self):
        n = len(self.ds)
        self.pairs = {}
        pbar = tqdm.tqdm_notebook(total=self.N)
        while(len(self.pairs) < self.N):
            i1, i2 = np.random.choice(n, size=2)
            y = w2dist(self.ys.loc[i1, 'mn'], self.ys.loc[i2, 'mn'])
            if (i1, i2) not in self.pairs.keys():
                self.pairs[(i1, i2)] = y
                pbar.update(1)
        pbar.close()
        self.id_to_pair = dict(zip(np.arange(self.N), list(self.pairs.keys())))
        
    def __getitem__(self, i):
        i1, i2 = self.id_to_pair[i]
        y = self.pairs[(i1, i2)]
        x1, x2 = self.ds[i1], self.ds[i2]
        return x1, x2, y