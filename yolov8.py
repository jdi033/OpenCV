from symtable import Class

import torch
import torch.nn as nn

#自动填充
#当需要输入通道数与输出通道数保持一致时，stride步长为1时，计算需要填充多少
def autopad(k, p=None, d=1):
    #d为空洞卷积，在卷积核的元素之间插入“空洞”，增加核的大小，从而扩大感受野，因此需要重新计算核的大小
    if d>1:
        # if isinstance(k, int)如果k为整数，else [d*(x+1)-1 for x in k]或者k为元组，则k中每个值都需重新计算
        k = d*(k+1)-1 if isinstance(k, int) else [d*(x+1)-1 for x in k]
    if p is None:
        p = k//2 if isinstance(k, int) else [x//2 for x in k]
    return p

#卷积
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

#Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=(3,3), s=1, p=None, add=True, e=0.5):
        super().__init__()
        c = int(c2*e)
        self.conv1 = Conv(c1, c, k[0], s, p)
        self.conv2 = Conv(c, c2, k[1], s, p)
        self.add = add and c1==c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

#C2f
class C2f(nn.Module):
    def __init__(self, c1, c2, s=1, p=None, n=1, add=True, e=0.5):
        super().__init__()
        self.c = int(c2*e)
        self.conv1 = Conv(c1, 2*self.c, 1, 1)
        self.conv2 = Conv((n+2)*self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, k=((3,3),(3,3)), s=s, p=p, add=add, e=1) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1//2
        self.conv1 = nn.Conv2d(c1, c_, 1, 1)
        self.conv2 = nn.Conv2d(c_*4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = self.conv1(x)
        y1 = self.m(y)
        y2 = self.m(y1)
        return self.cv2(torch.cat((y, y1, y2, self.m(y2)), 1))

class DFL(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        grid = torch.arange(reg_max, dtype=torch.float)
        self.register_buffer('grid', grid.view(1, reg_max, 1, 1))
        self.c1 = reg_max

    def forward(self, x):
        b, c, a = x.shape()
        x_reshape = x.view(b, 4, self.c1, a).permute(0, 2, 1, 3)
        prob_dist = x_reshape.softmax(1)
        return self.conv(prob_dist).view(b, 4, a)

class Detect(nn.Module):
    dynamic = False
    export = False
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.reg_output_dim = 4*self.reg_max
        self.no = nc + 4*self.reg_max
        self.stride = torch.zeros(len(ch))

        c2 = max(16, ch[0]//4, 4*self.reg_max)
        c3 = max(ch[0]//4, min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c2, 3),
                nn.Conv2d(c2, c2, 3),
                nn.Conv2d(c2, self.nc, 1),
            ) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c3, 3),
                nn.Conv2d(c3, c3, 3),
                nn.Conv2d(c3, 4*self.reg_max, 1),
            ) for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape()

        y_cls=[]
        y_reg=[]
        for i in range(self.nl):
            y_cls.append(self.cv2[i](x[i]))
            y_reg.append(self.cv3[i](x[i]))

        cls_concatenated = torch.cat(y_cls, 1)
        reg_concatenated = torch.cat(y_reg, 1)
        pred = torch.cat((cls_concatenated, reg_concatenated), 1)

        if self.training:
            return pred
        else:
            reg_to_decode, cls_results = pred.split((self.reg_output_dim, self.nc), 1)
            cls_scores = cls_results.softmax(1)
            reg_results = self.dfl(reg_to_decode)
            out = torch.cat((cls_scores, reg_results), 1)
            return out