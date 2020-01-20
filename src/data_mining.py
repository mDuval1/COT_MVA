import torch
import tqdm
import numpy as np
import ot
import matplotlib.pyplot as plt

from .utils import decode_latent


class DigitDataset(torch.utils.data.Dataset):
    
    def __init__(self, ds, units):
        self.i2j = {}
        self.labels = []
        self.units = units
        self.ds = ds
        i = 0
        for j in range(len(self.ds)):
            if self.ds[j][1] in self.units:
                self.labels.append(self.ds[j][1])
                self.i2j[i] = j
                i += 1
        self.n = i
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        im = self.ds[self.i2j[i]][0]
        im /= im.sum()
        return im


def get_avg_digit(ds):
    d = np.zeros(ds[0].shape)
    for i in range(len(ds)):
        d += ds[i].numpy()
    d /= len(ds)
    return d

def get_avg_digit_W(ds, encoder, decoder):
    imgs = torch.stack([ds[i] for i in range(len(ds))])
    mean_latent = encoder(imgs).mean(dim=0)[None]
    return decoder(mean_latent)[0].detach().numpy()


class WKmeans:
    
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        
    def _kmplusplusinit(self, k):
        centroid_ids = np.random.choice(self.n, size=1)
        self.centroids = self.latent[centroid_ids]
        self._compute_assignment()
        while len(centroid_ids) < k:
            distrib = self.dists2centroids[np.arange(self.n), self.assignment]**2
            p = distrib / distrib.sum()
            next_centroid = np.random.choice(self.n, size=1, p=p)
            centroid_ids = np.append(centroid_ids, next_centroid)
            self.centroids = self.latent[centroid_ids]
            self._compute_assignment()
            
    def _init_fit(self, k):
#         centroid_ids = np.random.choice(self.n, size=k, replace=False)
#         self.centroids = self.latent[centroid_ids]
#         self._compute_assignment()
        self._kmplusplusinit(k)
    
    def _compute_latent(self, dataset, has_y=False):
        latent = []
        for i in tqdm.tqdm_notebook(range(self.n), desc='Computing embeddings'):
            x = dataset[i]
            if has_y:
                x = x[0]
            latent.append(self.encoder(x[None]).detach().cpu().numpy()[0])
        self.latent = np.stack(latent)
        
    def _compute_assignment(self):
        self.dists2centroids = ot.dist(x1=self.latent, x2=self.centroids)
        self.assignment = self.dists2centroids.argmin(axis=1)
        
    def _compute_centroids(self):
        for i in range(len(self.centroids)):
            grad = self.latent[self.assignment == i].mean(axis=0)
            self.centroids[i] = (1 - self.lr) * self.centroids[i] + self.lr * grad
    
    def _compute_likelihood(self):
        return self.dists2centroids[np.arange(self.n), self.assignment].mean()
        
    def _iter(self):
        self._compute_centroids()
        self._compute_assignment()
        
    def fit(self, dataset, k, has_y=False, max_iter=100, tol=1e-6, lr=1e-3, ntries=10):
        self.n = len(dataset)
        self.lr = lr
        self._compute_latent(dataset, has_y=has_y)
        
        centroids = []
        self.Ls = np.zeros(ntries)
        for i in tqdm.tqdm_notebook(range(ntries)):
            self._init_fit(k)
            oldL = np.infty
            newL = self._compute_likelihood()
            current_iter = 0
#             pbar = tqdm.tqdm_notebook(desc='k-means iter')
            self.delta = np.abs(newL - oldL)
            while self.delta > tol and current_iter < max_iter:
                oldL = newL
                self._iter()
                newL = self._compute_likelihood()
                self.delta = np.abs(newL - oldL)
                current_iter += 1
            self.Ls[i] = newL
            centroids.append(self.centroids)
        self.centroids = centroids[np.argmin(self.Ls)]
        self._compute_assignment()
#                 pbar.update(1)
#             pbar.close()


def plot_clusters(ds, algo, decoder, m=9, random=False):
    dists = np.copy(algo.dists2centroids)
    k = len(algo.centroids)
    fig, axs = plt.subplots(figsize=(9, k+1), ncols=m+1, nrows=k)
    for i in range(k):
        im = decode_latent(decoder, algo.centroids[i])[0]
        axs[i, 0].imshow(im, cmap='Greens', interpolation='nearest')
        plt.setp(axs[i, 0].get_xticklabels(), visible=False)
        plt.setp(axs[i, 0].get_yticklabels(), visible=False)
        axs[i, 0].tick_params(axis='both', which='both', length=0)
        axs[i, 0].set_ylabel(f'centroid {i+1}')
        if random:
            chosen = np.random.choice(np.nonzero(algo.assignment == i)[0], size=m, replace=False)
        else:
            chosen = np.argsort(dists[:, i])[:m]
        for j in range(1, m+1):
            axs[i, j].imshow(ds[chosen[j-1]][0].numpy(), cmap='Greens', interpolation='nearest')
            plt.setp(axs[i, j].get_xticklabels(), visible=False)
            plt.setp(axs[i, j].get_yticklabels(), visible=False)
            axs[i, j].tick_params(axis='both', which='both', length=0)
    axs[0, 0].set_title('Centroids')
    word = 'random' if random else 'closest'
    axs[0, 1 + m//2].set_title(f'{m} {word} points of the cluster')
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.99, top=0.95, bottom=0.01)
    return fig, axs