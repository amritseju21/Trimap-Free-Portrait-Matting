import torch 
import torch.nn as nn

class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sample: str 
    ):
        super().__init__()
        if sample == 'down':
            self.resample = nn.MaxPool2d(2)
        elif sample == 'up':
            self.resample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.resample = nn.Identity()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.resample(h)
        
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        
        return h



class UNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # downsampling blocks
        self.in_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.d_block1 = Block(16, 32, sample=None)
        self.d_block2 = Block(32, 64, sample='down')
        self.d_block3 = Block(64, 128, sample='down')
        self.d_block4 = Block(128, 256, sample='down')

        # upsampling blocks
        self.u_block4 = Block(256, 256//2, sample='up')
        self.u_block3 = Block(256, 128//2, sample='up')
        self.u_block2 = Block(128, 64//2, sample='up')
        self.u_block1 = Block(64, 32//2, sample=None)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = images

        h = self.in_conv(h)
        h1 = self.d_block1(h)
        h2 = self.d_block2(h1)
        h3 = self.d_block3(h2)
        h4 = self.d_block4(h3)
        #h4 is bottleneck
        h = self.u_block4(h4)
        h = self.u_block3(torch.cat((h, h3), dim=1))
        h = self.u_block2(torch.cat((h, h2), dim=1))
        h = self.u_block1(torch.cat((h, h1), dim=1))
        h = self.out_conv(h)
        h = torch.sigmoid(h)

        return h