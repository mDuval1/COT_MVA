import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, input_dim, embed_dim, hidden_dims):
        super(Encoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        hin = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hin, h))
            hin = h
        layers.extend([nn.ReLU(), nn.Linear(hin, embed_dim), nn.ReLU()])
        self.f = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.f(x)

class Decoder(nn.Module):
    
    def __init__(self, embed_dim, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.sfm = nn.Softmax()
        layers = [nn.Linear(embed_dim, hidden_dims[0])]
        hin = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hin, h))
            hin = h
        layers.extend([nn.ReLU(), nn.Linear(hin, output_dim), nn.Softmax()])
        self.f = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.f(x)


class EncoderCNN(nn.Module):
    
    def __init__(self, latent_dim=50):
        super(EncoderCNN, self).__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(*[
            nn.Conv2d(1, 20, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(20, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(10, 5, 5, 1, 2),
            nn.ReLU()
        ])
        self.fc = nn.Sequential(*[
            nn.Linear(3920, 100),
            nn.ReLU(),
            nn.Linear(100, self.latent_dim),
            nn.ReLU()
        ])
        
    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), start_dim=1))


class DecoderCNN(nn.Module):
    
    def __init__(self, latent_dim=50):
        super(DecoderCNN, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(*[
            nn.Linear(self.latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 3920),
            nn.ReLU()
        ])
        self.conv = nn.Sequential(*[
            nn.Conv2d(5, 10, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(20, 1, 3, 1, 1),
        ])
        self.sfm = nn.Softmax()
        
        
    def forward(self, x):
        x = self.fc(x)
        x = self.conv(torch.reshape(x, (x.shape[0], 5, 28, 28)))
        x = self.sfm(x.view(x.shape[0], -1))
        x = x.view(x.shape[0], 1, 28, 28)
        return x