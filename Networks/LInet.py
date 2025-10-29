import torch
import torch.nn as nn

class LIRB(nn.Module):  # Low-Illumination Residual Block
    def __init__(self, in_channels, out_channels):
        super(LIRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.match_channels:
            residual = self.match_channels(residual)
        out += residual
        out = self.relu(out)
        return out

class CLA(nn.Module):  # Channel-Level Adapter
    def __init__(self, in_channels, out_channels):
        super(CLA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        return self.conv(x)

class LCP(nn.Module):  # Low-light Channel Processor
    def __init__(self):
        super(LCP, self).__init__()

        self.initial = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.resblock1 = LIRB(32, 32)
        self.resblock2 = LIRB(32, 64)
        self.resblock3 = LIRB(64, 64)
        self.resblock4 = LIRB(64, 128)
        self.resblock5 = LIRB(128, 64)
        self.resblock6 = LIRB(64, 64)
        self.resblock7 = LIRB(64, 32)
        self.resblock8 = LIRB(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.adjust1 = CLA(32, 64)
        self.adjust2 = CLA(64, 128)
        self.adjust3 = CLA(64, 32)
        
    def forward(self, x):
        x_initial = self.initial(x)
        x1 = self.resblock1(x_initial)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2 + self.adjust1(x1))
        x4 = self.resblock4(x3)
        x5 = self.resblock5(x4 + self.adjust2(x3))
        x6 = self.resblock6(x5)
        x7 = self.resblock7(x6 + x5)  
        x8 = self.resblock8(x7)
        x_final = self.final(x8 + x7)

        return x_final

class LINet(nn.Module):  # Low-Illumination Network
    def __init__(self):
        super(LINet, self).__init__()
        
        self.red = LCP()
        self.green = LCP()
        self.blue = LCP()
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r = self.red(x[:, 0:1, :, :])
        g = self.green(x[:, 1:2, :, :])
        b = self.blue(x[:, 2:3, :, :])
        combined = torch.cat([r, g, b], dim=1)
        x = self.sigmoid(self.final_conv(combined))
        return x


