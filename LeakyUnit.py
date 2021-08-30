import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyUnit(nn.Module):
    def __init__(self, n_features):
        super(LeakyUnit, self).__init__()
        self.W_r = nn.Conv2d(2*n_features, n_features, kernel_size=1, padding=0, stride=1, bias=False)
        self.W = nn.Conv2d(n_features, n_features, kernel_size=1, padding=0, stride=1, bias=False)
        self.U = nn.Conv2d(n_features, n_features, kernel_size=1, padding=0, stride=1, bias=False)
        self.W_z = nn.Conv2d(2*n_features, n_features, kernel_size=1, padding=0, stride=1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, f_m, f_n):
        f_mn = torch.cat((f_m, f_n), dim=1)
        r_mn = self.sigma(self.W_r(f_mn))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn.expand_as(f_n) * f_n))
        z_mn = self.sigma(self.W_z(f_mn))
        f_m_out = z_mn.expand_as(f_m) * f_m + (1 - z_mn.expand_as(f_mn_hat)) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n

        return f_m_out, r_mn, z_mn
