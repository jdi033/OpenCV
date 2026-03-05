from symtable import Class

import torch
import torch.nn as nn

#自动填充
def autopad(k, p=None, d=1):
    if d>1:
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