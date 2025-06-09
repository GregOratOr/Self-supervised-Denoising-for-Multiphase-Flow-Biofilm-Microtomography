import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, n_ip, n_out, kernel_size=3, gain=torch.sqrt(torch.tensor(2, dtype=torch.float32)), bias=True):
        super(ConvLayer, self).__init__()

        wstd = gain / torch.sqrt(torch.prod(torch.tensor([kernel_size, kernel_size , n_ip], dtype=torch.float32)))
        self.conv = nn.Conv2d(in_channels=n_ip, out_channels=n_out, kernel_size=kernel_size, padding=kernel_size//2, stride=1, bias=bias)
        # Initialize weights
        nn.init.normal_(self.conv.weight, mean=0.0, std=wstd)
        # Initialize Bias
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)
    
class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # Tensor structure: [batch_size, channels, height, width]
        # return x.repeat_interleave(self.scale_factor, dim=2).repeat_interleave(self.scale_factor, dim=3) # tile across dim=2 (height) and 3(width)
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class DownsampleLayer(nn.Module):
    def __init__(self, scale_factor=2):
        super(DownsampleLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor, padding=0)

    def forward(self, x):
        return self.pool(x)
    
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        
    def forward(self, layers):
        return torch.cat(layers, dim=1)
        
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.1):
        super(LeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return self.leaky_relu(x)
        
class Noise2Noise(nn.Module):
    def __init__(self):
        super(Noise2Noise, self).__init__()
        self.enc_conv0 = ConvLayer(1, 48)
        self.enc_conv1 = ConvLayer(48, 48)
        self.pool1 = DownsampleLayer() # 24
        
        self.enc_conv2 = ConvLayer(48, 48)
        self.pool2 = DownsampleLayer() # 24
        
        self.enc_conv3 = ConvLayer(48, 48)
        self.pool3 = DownsampleLayer() # 24
        
        self.enc_conv4 = ConvLayer(48, 48)
        self.pool4 = DownsampleLayer() # 24
        
        self.enc_conv5 = ConvLayer(48, 48)
        self.pool5 = DownsampleLayer() # 24
        
        self.enc_conv6 = ConvLayer(48, 48)
        
        self.upsample5 = UpsampleLayer() # 48
        self.concat5 = Concat()
        self.dec_conv5 = ConvLayer(96, 96)
        self.dec_conv5b = ConvLayer(96, 96)
        
        self.upsample4 = UpsampleLayer() # 96
        self.concat4 = Concat()
        self.dec_conv4 = ConvLayer(144, 96)
        self.dec_conv4b = ConvLayer(96, 96)
        
        self.upsample3 = UpsampleLayer() # 96
        self.concat3 = Concat()
        self.dec_conv3 = ConvLayer(144, 96)
        self.dec_conv3b = ConvLayer(96, 96)
        
        self.upsample2 = UpsampleLayer() # 96
        self.concat2 = Concat()
        self.dec_conv2 = ConvLayer(144, 96)
        self.dec_conv2b = ConvLayer(96, 96)
        
        self.upsample1 = UpsampleLayer() # 96
        self.concat1 = Concat()
        self.dec_conv1a = ConvLayer(96 + 1, 64)
        self.dec_conv1b = ConvLayer(64, 32)
        
        self.dec_conv1 = ConvLayer(32, 1, gain=1.0)
            
    def forward(self, x):
        if int(x.shape[-1])%2 == 1:
            x = F.pad(x, (0, 1, 0, 1), "constant", -0.5)
        
        # x = F.pad(x, (0, 1, 0, 1), "constant", -0.5)
        input = x.unsqueeze(1)
        
        x = F.leaky_relu(self.enc_conv0(input), 0.1)
        x = F.leaky_relu(self.enc_conv1(x), 0.1)
        x = self.pool1(x)
        pool1 = x
        
        x = F.leaky_relu(self.enc_conv2(pool1), 0.1)
        x = self.pool2(x)
        pool2 = x
        
        x = F.leaky_relu(self.enc_conv3(pool2), 0.1)
        x = self.pool3(x)
        pool3 = x
        
        x = F.leaky_relu(self.enc_conv4(pool3), 0.1)
        x = self.pool4(x)
        pool4 = x
        
        x = F.leaky_relu(self.enc_conv5(pool4), 0.1)
        x = self.pool5(x)
        pool5 = x
        x = F.leaky_relu(self.enc_conv6(x), 0.1)
        
        x = self.upsample5(x)
        x = self.concat5([x, pool4])
        x = F.leaky_relu(self.dec_conv5(x), 0.1)
        x = F.leaky_relu(self.dec_conv5b(x), 0.1)
        
        x = self.upsample4(x)
        x = self.concat4([x, pool3])
        x = F.leaky_relu(self.dec_conv4(x), 0.1)
        x = F.leaky_relu(self.dec_conv4b(x), 0.1)
        
        x = self.upsample3(x)
        x = self.concat3([x, pool2])
        x = F.leaky_relu(self.dec_conv3(x), 0.1)
        x = F.leaky_relu(self.dec_conv3b(x), 0.1)
        
        x = self.upsample2(x)
        x = self.concat2([x, pool1])
        x = F.leaky_relu(self.dec_conv2(x), 0.1)
        x = F.leaky_relu(self.dec_conv2b(x), 0.1)
        
        x = self.upsample1(x)
        x = self.concat1([x, input])
        x = F.leaky_relu(self.dec_conv1a(x), 0.1)
        x = F.leaky_relu(self.dec_conv1b(x), 0.1)
        
        x = self.dec_conv1(x)
        return x.squeeze(1)[:, :-1, :-1]