import functools
import numpy as np
import pandas as pd
import ot
import torch
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import tqdm



def process(im):
    im = np.array(im.flatten(), dtype=np.float64)
    im /= im.sum()
    return im

def compute_dist(im1, im2, M):
    return ot.emd2(process(im1), process(im2), M)


def decode_latent(decoder, z):
    return decoder(torch.tensor(z).float()[None]).detach().cpu().numpy()[0]


def partial_dist(p):
    return functools.partial(scipy.spatial.minkowski_distance_p, p=2)


def w2dist(mn1, mn2):
    """Computes the Wasserstein W_2 squared distance for two multivariate
    scipy gaussians.
    """
    mean_dist = np.linalg.norm(mn1.mean - mn2.mean)**2
    mn1covhalf = scipy.linalg.fractional_matrix_power(mn1.cov, 0.5)
    M = scipy.linalg.fractional_matrix_power(mn1covhalf @ mn2.cov @ mn1covhalf, 0.5)
    trM = mn1.cov + mn2.cov - 2 * M
    dsquared = mean_dist + np.trace(trM)
    return dsquared


def generate_dataset(N, limits, sample_size=100, bins=np.linspace(-10, 10, 100)):
    ds = []
    ys = []
    c = 0.5
    min_mu, max_mu, min_sigma, max_sigma = limits
    for i in tqdm.tqdm_notebook(range(N)):
        mu = min_mu + scipy.stats.beta.rvs(a=c, b=c) * (max_mu - min_mu)
        sigma = min_sigma + scipy.stats.beta.rvs(a=c, b=c) * (max_sigma - min_sigma)
        mn = scipy.stats.multivariate_normal(mean=mu, cov=sigma**2)
        x = mn.pdf(bins)
        x = x / x.sum()
        ds.append(x)
        ys.append({'mu': mu, 'sigma': sigma, 'mn': mn})
    return np.stack(ds), pd.DataFrame(ys)