import torch
import torch.nn as nn
import torchvision.models as models
from LiCNN.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2


def bgr2gray(bgr):

    r, g, b = bgr[:, [2], :, :], bgr[:, [1], :, :], bgr[:, [0], :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class cls_criterion(nn.Module):
    def __init__(self):
        super(cls_criterion, self).__init__()
        self.loss_fc = nn.CrossEntropyLoss()

    def forward(self, output, target):
        loss = self.loss_fc(output, target)
        return loss


class tsf_criterion(nn.Module):
    def __init__(self):
        super(tsf_criterion, self).__init__()
        self.loss_fc = nn.L1Loss()

    def forward(self, output, target):
        loss = self.loss_fc(output, target)  #.div_(output.view(-1).size(0))
        return loss


class rec_criterion(nn.Module):
    def __init__(self):
        super(rec_criterion, self).__init__()
        self.light_cnn = LightCNN_9Layers()
        checkpoint = torch.load("./LiCNN/LightCNN_9Layers_checkpoint.pth.tar")["state_dict"]
#         self.light_cnn = LightCNN_29Layers()
#         checkpoint = torch.load("./LiCNN/LightCNN_29Layers_checkpoint.pth.tar")["state_dict"]
#         self.light_cnn = LightCNN_29Layers_v2()
#         checkpoint = torch.load("./LiCNN/LightCNN_29Layers_v2_checkpoint.pth.tar")["state_dict"]
        
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        self.light_cnn.load_state_dict(checkpoint)
        self.light_cnn.eval()
        for param in self.light_cnn.parameters():
            param.requires_grad = False
        self.feature_dis = nn.MSELoss()
        self.image_dis = nn.L1Loss()

    def forward(self, output, target):
        if output.size(1) == 3:
            output = bgr2gray(output)
            target = bgr2gray(target)
        feature_out = self.light_cnn(output)
        feature_tar = self.light_cnn(target)
        loss_feature = self.feature_dis(feature_out, feature_tar)
        loss_img_dis = self.image_dis(output, target)
        return 0.05 * loss_feature + loss_img_dis  #


class dcn_criterion(nn.Module):
    def __init__(self):
        super(dcn_criterion, self).__init__()
        self.loss_fc = nn.MSELoss()

    def forward(self, output, target):
        loss = self.loss_fc(output, target)
        return loss
