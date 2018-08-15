import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

def conv7x7(in_channels,out_channels,stride=1,padding=3,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=7,stride=stride,padding=padding,bias=bias,groups=groups)
def conv3x3(in_channels,out_channels,stride=1,padding=1,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=bias,groups=groups)
def conv1x1(in_channels,out_channels,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=True,groups=groups)

class PrimaryModule(nn.Module):
    def __init__(self,in_channels=3,out_channels=32):
        super(PrimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.PrimaryModule = nn.Sequential(
                                           OrderedDict(
                                                       [
                                                        ('PrimaryBN',nn.BatchNorm2d(in_channels)),
                                                        ('PrimaryConv7x7',conv7x7(in_channels,out_channels,2,3,True,1)),
                                                        ('PrimaryConv7x7BN',nn.BatchNorm2d(out_channels)),
                                                        ('PrimaryConv7x7ReLU',nn.ReLU())
                                                       ]
                                                       )
                                           )
    def forward(self,x):
        x = self.PrimaryModule(x)
        return x

class BlockCompress(nn.Module):
    def __init__(self,in_channels=32,growk=32):
        super(BlockCompress, self).__init__()
        self.in_channels = in_channels
        self.growk = growk
        
        self.BlockCompress = nn.Sequential(
                                           OrderedDict(
                                                       [
                                                        ('conv1x1BN',nn.BatchNorm2d(in_channels)),
                                                        ('conv1x1ReLU',nn.ReLU()),
                                                        ('conv1x1',conv1x1(in_channels,4*growk,True,1)),
                                                        ('conv3x3BN',nn.BatchNorm2d(4*growk)),
                                                        ('conv3x3ReLU',nn.ReLU()),
                                                        ('conv3x3',conv3x3(4*growk,growk,1,1,True,1))
                                                       ]
                                                       )
                                           )
    def forward(self,x):
        x = self.BlockCompress(x)
        return x

class DenseNetBlock(nn.Module):
    def __init__(self,in_channels=32,growk=32,layers=6):
        super(DenseNetBlock, self).__init__()
        self.in_channels = in_channels
        self.growk = growk
        self.layers = layers
        
        self.DenseNetBlock = nn.ModuleList()
        for i in range(0,layers):
            self.DenseNetBlock.append(BlockCompress(i*growk+in_channels,growk))
    def forward(self,x):
        features = [x]
        for i,l in enumerate(self.DenseNetBlock):
            x = l(torch.cat(features,dim=1))
            features.append(x)
        x = torch.cat(features,dim=1)
        return x

class CompetitiveSE(nn.Module):
    def __init__(self,in_channels=32):
        super(CompetitiveSE, self).__init__()
        self.in_channels = in_channels
        
        self.Compress = nn.Sequential(
                                      OrderedDict(
                                                  [
                                                   ('CompressReLU',nn.ReLU()),
                                                   ('Compressconv1x1',conv1x1(2*in_channels,in_channels//3,True,1))
                                                  ]
                                                  )
                                      )
        self.unCompress = nn.Sequential(
                                        OrderedDict(
                                                    [
                                                     ('unCompressReLU',nn.ReLU()),
                                                     ('unCompressconv1x1',conv1x1(in_channels//3,in_channels,True,1))
                                                    ]
                                                    )
                                        )
    def forward(self,x1,x2):
        weight_x1 = torch.mean(x1,dim=2,keepdim=True)
        weight_x1 = torch.mean(weight_x1,dim=3,keepdim=True)
        weight_x2 = torch.mean(x2,dim=2,keepdim=True)
        weight_x2 = torch.mean(weight_x2,dim=3,keepdim=True)
        weight = torch.cat([weight_x1,weight_x2],dim=1)
        weight = self.Compress(weight)
        weight = self.unCompress(weight)
        return weight

class PoolAttention(nn.Module):
    def __init__(self,in_channels,isPrimary=False):
        super(PoolAttention, self).__init__()
        self.in_channels = in_channels
        self.isPrimary = isPrimary
        
        self.LeftMaxPool = nn.Sequential(
                                         OrderedDict(
                                                     [
                                                      ('LeftMaxPool',nn.MaxPool2d((3,3),stride=(2,2),ceil_mode=True))
                                                     ]
                                                     )
                                         )
        self.LeftAvgPool = nn.Sequential(
                                         OrderedDict(
                                                     [
                                                      ('LeftAvgPool',nn.AvgPool2d((2,2),stride=(2,2),ceil_mode=True))
                                                     ]
                                                     )
                                         )
        self.RightF = nn.Sequential(
                                    OrderedDict(
                                                [
                                                 ('RightFBN',nn.BatchNorm2d(in_channels)),
                                                 ('RightFReLU',nn.ReLU()),
                                                 ('RightFconv3x3',conv3x3(in_channels,in_channels,2,1,True,1))
                                                ]
                                                )
                                    )
        self.RightG = nn.Sequential(
                                    OrderedDict(
                                                [
                                                 ('RightGBN',nn.BatchNorm2d(in_channels)),
                                                 ('RightGReLU',nn.ReLU()),
                                                 ('RightGconv3x3',conv3x3(in_channels,in_channels,2,1,True,1))
                                                ]
                                                )
                                    )
        self.RightH = nn.Sequential(
                                    OrderedDict(
                                                [
                                                 ('RightHBN',nn.BatchNorm2d(in_channels)),
                                                 ('RightHReLU',nn.ReLU()),
                                                 ('RightHconv3x3',conv3x3(in_channels,in_channels,2,1,True,1))
                                                ]
                                                )
                                    )
        self.Softmax = nn.Softmax(dim=2)
        if torch.cuda.is_available():
            self.gamma = torch.zeros((1),requires_grad=True).cuda(0)
        else:
            self.gamma = torch.zeros((1),requires_grad=True)
        self.CompetitiveSE = CompetitiveSE(in_channels)
    def forward(self,x):
        if self.isPrimary:
            left = self.LeftMaxPool(x)
        else:
            left = self.LeftAvgPool(x)
        rightf = self.RightF(x)
        rightg = self.RightG(x)
        righth = self.RightH(x)
        s = torch.matmul(rightf.view((rightf.size(0),rightf.size(1),-1)).permute(0,2,1),rightg.view((rightg.size(0),rightg.size(1),-1)))
        s = self.Softmax(s)
        s = torch.matmul(righth.view(righth.size(0),righth.size(1),-1),s).view(righth.size())
        s = self.gamma*s
        
        se = self.CompetitiveSE(x,s)
        x = (left+s)*(1.0+se)
        return x

class TransitionModule(nn.Module):
    def __init__(self,in_channels,isPrimary=False):
        super(TransitionModule ,self).__init__()
        self.in_channels = in_channels
        self.isPrimary = isPrimary
        
        self.Compress = nn.Sequential(
                                      OrderedDict(
                                                  [
                                                   ('CompressBN',nn.BatchNorm2d(in_channels)),
                                                   ('CompressReLU',nn.ReLU()),
                                                   ('Compressconv1x1',conv1x1(in_channels,in_channels//2,True,1))
                                                  ]
                                                  )
                                      )
        self.Attention = PoolAttention(in_channels//2,isPrimary)
    def forward(self,x):
        x = self.Compress(x)
        x = self.Attention(x)
        return x

class FinalModule(nn.Module):
    def __init__(self,in_channels=960,num_classes=1000):
        super(FinalModule, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.FC = nn.Sequential(
                                OrderedDict(
                                            [
                                             ('Dropout',nn.Dropout(0.5)),
                                             ('FC',conv1x1(in_channels,num_classes,True,1))
                                            ]
                                            )
                                )
    def forward(self,x):
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = self.FC(x)
        return x

class PADenseNet(nn.Module):
    def __init__(self,in_channels,num_classes=1000,growk=12):
        super(PADenseNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.growk = growk
        self.layers = [6,12,24,16]
        self.out_channels = [(growk,growk)]
        for i,_layers in enumerate(self.layers):
            self.out_channels.append( (self.out_channels[i][1]+_layers*growk,(self.out_channels[i][1]+_layers*growk)//2) )
        
        self.PrimaryModule = PrimaryModule(in_channels,self.out_channels[0][0])
        self.PrimaryPA = PoolAttention(self.out_channels[0][1],True)
        
        self.Block1 = DenseNetBlock(self.out_channels[0][1],growk,self.layers[0])
        self.Transition1 = TransitionModule(self.out_channels[1][0])
        
        self.Block2 = DenseNetBlock(self.out_channels[1][1],growk,self.layers[1])
        self.Transition2 = TransitionModule(self.out_channels[2][0])
        
        self.Block3 = DenseNetBlock(self.out_channels[2][1],growk,self.layers[2])
        self.Transition3 = TransitionModule(self.out_channels[3][0])
        
        self.Block4 = DenseNetBlock(self.out_channels[3][1],growk,self.layers[3])
        
        self.FC = FinalModule(self.out_channels[4][0],num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    def forward(self,x):
        x = self.PrimaryModule(x)
        x = self.PrimaryPA(x)
        
        x = self.Block1(x)
        x = self.Transition1(x)
        
        x = self.Block2(x)
        x = self.Transition2(x)
        
        x = self.Block3(x)
        x = self.Transition3(x)
        
        x = self.Block4(x)
        x = self.FC(x)
        
        x = x.view((x.size(0),x.size(1)))
        return x

if __name__=='__main__':
    import numpy as np
    net = PADenseNet(3,1000,12)
    net.train()
    input = torch.randn(1,3,224,224)
    output = net(input)
    print(output.detach().numpy().shape)