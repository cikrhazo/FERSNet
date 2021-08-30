import torch
import torch.nn as nn
from LeakyUnit import LeakyUnit
from net_utlz.blocks import BasicBlock, conv1x1, UNetUp, Transform

cfg = {
    'VGG13': [(64, 64), (64, 64), 'D1:128, 64',
              (128, 128), (128, 128), 'D:128, 64',
              (256, 128), (256, 256), 'D:256, 128',
              (512, 256), (512, 512), 'D:512, 256',
              (512, 512), (1024, 512), 'D:512, 512'],
}


class Decoder(nn.Module):
    def __init__(self, vgg_name='VGG13', out_channel=3, mem=512, num_class=6):
        super(Decoder, self).__init__()
        self.module = self.make_layers(cfg[vgg_name])
        self.out_layer = nn.Conv2d(64, out_channel, 3, 1, 1)
        self.transform = Transform(in_channel=mem, style_dim=num_class)

    def forward(self, feature, prob_t, shortcut_list, transform=False):
        shortcut_list = shortcut_list[::-1]
        if transform:
            y = self.transform(feature, prob_t)
        else:
            y = feature
        k = 0
        for operation in self.module:
            if isinstance(operation, UNetUp):
                shortcut = shortcut_list[k]
                y = operation(y, shortcut, prob_t)
                k = k + 1
            else:
                y = operation(y)
        return self.out_layer(y)

    def make_layers(self, cfg):
        layers = []
        for x in cfg:
            if 'D1' in x:
                chs = x.split(":")[-1]
                in_ch, out_ch = int(chs.split(",")[0]), int(chs.split(",")[1])
                layers += [nn.LeakyReLU(0.2),
                           nn.InstanceNorm2d(out_ch),
                           nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
            elif 'D' in x:
                chs = x.split(":")[-1]
                in_ch, out_ch = int(chs.split(",")[0]), int(chs.split(",")[1])
                layers += [UNetUp(in_ch, out_ch)]
            else:
                in_ch, out_ch = x[0], x[1]
                layers += [nn.LeakyReLU(0.2),
                           nn.InstanceNorm2d(out_ch),
                           nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        layers = layers[::-1]
        return nn.ModuleList(layers)


class Classifier(nn.Module):
    def __init__(self, nz=7, nc=512, _size=1):
        super(Classifier, self).__init__()
        self.pooling = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.BatchNorm2d(nc),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(_size)
        )
        self.classifier = nn.Linear(nc, nz)

    def forward(self, feature):
        pooled = self.pooling(feature).view(feature.size(0), -1)
        out = self.classifier(pooled)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape=(1, 96, 96), num_class=6):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.out_size = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels + num_class, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, img, cond):
        img_input = torch.cat((img, cond), 1)
        dis_out = self.model(img_input)
        return dis_out


class FERSNet(nn.Module):
    def __init__(self, vgg_name='VGG13', num_class=6, mem_size=512, k_channel=1, inter=64):
        super(FERSNet, self).__init__()
        self.leakyunitxy1 = LeakyUnit(n_features=inter * 2)
        self.leakyunityx1 = LeakyUnit(n_features=inter * 2)
        self.leakyunitxy2 = LeakyUnit(n_features=inter * 4)
        self.leakyunityx2 = LeakyUnit(n_features=inter * 4)
        self.leakyunitxy3 = LeakyUnit(n_features=inter * 8)
        self.leakyunityx3 = LeakyUnit(n_features=inter * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(k_channel, inter, 3, 1, 1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        self.conv1x = BasicBlock(
            inplanes=inter, planes=2 * inter, downsample=conv1x1(inter, 2 * inter)
        )
        self.conv1y = BasicBlock(
            inplanes=inter, planes=2 * inter, downsample=conv1x1(inter, 2 * inter), norm_layer=nn.InstanceNorm2d
        )

        self.conv2x = BasicBlock(
            inplanes=2 * inter, planes=4 * inter, downsample=conv1x1(2 * inter, 4 * inter)
        )
        self.conv2y = BasicBlock(
            inplanes=2 * inter, planes=4 * inter, downsample=conv1x1(2 * inter, 4 * inter), norm_layer=nn.InstanceNorm2d
        )

        self.conv3x = BasicBlock(
            inplanes=4 * inter, planes=8 * inter, downsample=conv1x1(4 * inter, 8 * inter)
        )
        self.conv3y = BasicBlock(
            inplanes=4 * inter, planes=8 * inter, downsample=conv1x1(4 * inter, 8 * inter), norm_layer=nn.InstanceNorm2d
        )

        self.conv4xy = BasicBlock(inplanes=16 * inter, planes=16 * inter)

        self.classifier = Classifier(nz=num_class, nc=512, _size=1)
        self.decoder = Decoder(vgg_name=vgg_name, out_channel=k_channel, mem=mem_size, num_class=num_class)
        self.pooling = nn.MaxPool2d(stride=2, kernel_size=2)

    def forward(self, x, prob_t):
        f1 = self.pooling(self.stem(x))
        shortcut = [f1]
        f_x2, f_y2 = self.pooling(self.conv1x(f1)), self.pooling(self.conv1y(f1))
        f_x2_hat, r_xy2, z_xy2 = self.leakyunitxy1(f_x2, f_y2)
        f_y2_hat, r_yx2, z_yx2 = self.leakyunityx1(f_y2, f_x2)
        shortcut.append(f_y2_hat)

        f_x3, f_y3 = self.pooling(self.conv2x(f_x2_hat)), self.pooling(self.conv2y(f_y2_hat))
        f_x3_hat, r_xy3, z_xy3 = self.leakyunitxy2(f_x3, f_y3)
        f_y3_hat, r_yx3, z_yx3 = self.leakyunityx2(f_y3, f_x3)
        shortcut.append(f_y3_hat)

        f_x4, f_y4 = self.pooling(self.conv3x(f_x3_hat)), self.pooling(self.conv3y(f_y3_hat))
        f_x4_hat, r_xy4, z_xy4 = self.leakyunitxy3(f_x4, f_y4)
        f_y4_hat, r_yx4, z_yx4 = self.leakyunityx3(f_y4, f_x4)
        shortcut.append(f_y4_hat)

        joint = torch.cat((f_x4_hat, f_y4_hat), dim=1)
        f_xy5 = self.pooling(self.conv4xy(joint))

        f_x5, f_y5 = f_xy5[:, :512], f_xy5[:, 512:]
        prob5 = self.classifier(f_x5)

        out = self.decoder(f_y5, prob_t, shortcut, transform=True)
        # out = self.decoder(f_xy5, prob_t, shortcut, transform=False)

        return out, prob5


if __name__ == '__main__':
    net = FERSNet(k_channel=3)
    net.cuda()
    dis = Discriminator()
    dis.cuda()

    inp = torch.randn(2, 3, 96, 96)
    inp = inp.cuda()
    prob_t = torch.randn(2, 6)
    prob_t = prob_t.cuda()

    out_tran, logits = net(inp, prob_t)

    print(out_tran.size())
    print(logits.size())
