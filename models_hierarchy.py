import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

class StatisticsNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim + z_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.layers(x)

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 128, dropout = False, n_class = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_class)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class ResNetMTL_InfoMin_CLUB(nn.Module):
    # for two level 
    def __init__(self, n_class, freeze=False, pretrained=False, dropout=False, max_length = 40):
        super().__init__()
        self.freeze = freeze
        
        self.n_class = n_class # [3,9,12]

        base_model = torchvision.models.resnet18(pretrained=pretrained)
        base_layers = list(base_model.children())
        
        self.base_bone = nn.Sequential(*base_layers[:9])
        
        size = [512, 256, 128, 64]
        
        fc1 = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Linear(size[1], 2)]
        self.cls_instance = nn.Sequential(*fc1)
        
        fc_h1 = [nn.Linear(size[0], size[1])]
        self.fc_h1 = nn.Sequential(*fc_h1)
        
        fc_h2 = [nn.Linear(size[0], size[1])]
        self.fc_h2 = nn.Sequential(*fc_h2)
        
        fc_shared = [nn.Linear(size[0], size[1])]
        self.fc_shared = nn.Sequential(*fc_shared)
        
        attention_net = [Attn_Net_Gated(L = size[2], D = size[3], dropout = dropout, n_class = n_class[-1])]
        self.attention_net = nn.Sequential(*attention_net)

        cls_patient1 = [nn.Linear(size[1], 2) for i in range(n_class[0])]
        self.cls_patient1 = nn.ModuleList(cls_patient1)
        
        cls_patient2 = [nn.Linear(size[1], 2) for i in range(n_class[1])]
        self.cls_patient2 = nn.ModuleList(cls_patient2)
        
        self.max_length = max_length
        
    def test(self, input, T_matrix):
        x = self.base_bone(input)

        x = torch.flatten(x, 1)
        
        out_instance = self.cls_instance(x)

        # x 50x512
        x1 = self.fc_h1(x)
        x2 = self.fc_h2(x)
        x_shared = self.fc_shared(x)
        
        h_size = int(x1.shape[-1]/2)

        x1_mu, x1_sigma = x1[...,:h_size], softplus(x1[...,h_size:])+1e-7
        x2_mu, x2_sigma = x2[...,:h_size], softplus(x2[...,h_size:])+1e-7
        x_shared_mu, x_shared_sigma = x_shared[...,:h_size], softplus(x_shared[...,h_size:])+1e-7
        
        dis1 = Independent(Normal(loc=x1_mu, scale=x1_sigma), 1)
        z1 = x1_mu
        
        dis2 = Independent(Normal(loc=x2_mu, scale=x2_sigma), 1)
        z2 = x2_mu
        
        dis_shared = Independent(Normal(loc=x_shared_mu, scale=x_shared_sigma), 1)
        z_shared = x_shared_mu
        
        A2, z_shared = self.attention_net(z_shared) # A: 40 x Classes    
        A2 = torch.transpose(A2, 1, 0)
        A2_o = A2
        A2 = F.softmax(A2, dim=1)
        A1 = torch.mm(T_matrix[-1], A2)
        
        M_h1 = torch.mm(A1, torch.cat((z1, z_shared), 1))
        M_h2 = torch.mm(A2, torch.cat((z2, z_shared), 1))# M: Classes x 256
        
        logits_h1 = torch.empty(self.n_class[0], 2).float().to('cuda')
        for c in range(self.n_class[0]):
            logits_h1[c] = self.cls_patient1[c](M_h1[c])

        
        logits_h2 = torch.empty(self.n_class[1], 2).float().to('cuda')
        for c in range(self.n_class[1]):
            logits_h2[c] = self.cls_patient2[c](M_h2[c])
            
        Mu = torch.cat((x1_mu, x2_mu, x_shared_mu), 1) # 50x384
        Sigma = torch.cat((x1_sigma, x2_sigma, x_shared_sigma), 1) # 50x384

        return out_instance, [Mu, Sigma, x1_mu, x2_mu, x_shared_mu, z1, z2, dis1, dis2,A1,A2,M_h1,M_h2], [logits_h1, logits_h2]

    def forward(self, input, T_matrix):
        x = self.base_bone(input)

        x = torch.flatten(x, 1)
        
        out_instance = self.cls_instance(x)

        # x 50x512
        x1 = self.fc_h1(x)
        x2 = self.fc_h2(x)
        x_shared = self.fc_shared(x)
        
        h_size = int(x1.shape[-1]/2)

        x1_mu, x1_sigma = x1[...,:h_size], softplus(x1[...,h_size:])+1e-7
        x2_mu, x2_sigma = x2[...,:h_size], softplus(x2[...,h_size:])+1e-7
        x_shared_mu, x_shared_sigma = x_shared[...,:h_size], softplus(x_shared[...,h_size:])+1e-7
        
        dis1 = Independent(Normal(loc=x1_mu, scale=x1_sigma), 1)
        z1 = dis1.rsample()
        
        dis2 = Independent(Normal(loc=x2_mu, scale=x2_sigma), 1)
        z2 = dis2.rsample()
        
        dis_shared = Independent(Normal(loc=x_shared_mu, scale=x_shared_sigma), 1)
        z_shared = dis_shared.rsample()
        
        A2, z_shared = self.attention_net(z_shared) # A: 40 x Classes    
        A2 = torch.transpose(A2, 1, 0)
        A2 = F.softmax(A2, dim=1)
        A1 = torch.mm(T_matrix[-1], A2)
        
        M_h1 = torch.mm(A1, torch.cat((z1, z_shared), 1))
        M_h2 = torch.mm(A2, torch.cat((z2, z_shared), 1))# M: Classes x 256
        
        logits_h1 = torch.empty(self.n_class[0], 2).float().to('cuda')
        for c in range(self.n_class[0]):
            logits_h1[c] = self.cls_patient1[c](M_h1[c])

        
        logits_h2 = torch.empty(self.n_class[1], 2).float().to('cuda')
        for c in range(self.n_class[1]):
            logits_h2[c] = self.cls_patient2[c](M_h2[c])
            
        Mu = torch.cat((x1_mu, x2_mu, x_shared_mu), 1) # 50x384
        Sigma = torch.cat((x1_sigma, x2_sigma, x_shared_sigma), 1) # 50x384

        return out_instance, [Mu, Sigma, x1_mu, x2_mu, x_shared_mu, z1, z2, dis1, dis2,A1,A2], [logits_h1, logits_h2]