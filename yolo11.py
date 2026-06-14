import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#自动填充
#当需要输入通道数与输出通道数保持一致时，stride步长为1时，计算需要填充多少
def autopad(k, p=None, d=1):
    #d为空洞卷积，在卷积核的元素之间插入“空洞”，增加核的大小，从而扩大感受野，因此需要重新计算核的大小
    if d>1:
        # if isinstance(k, int)如果k为整数，else [d*(x+1)-1 for x in k]或者k为元组，则k中每个值都需重新计算
        k = d*(k+1)-1 if isinstance(k, int) else [d*(x+1)-1 for x in k]
    #根据核k的大小，确定填充
    if p is None:
        p = k//2 if isinstance(k, int) else [x//2 for x in k]
    return p

#卷积
class Conv(nn.Module):
    #激活函数
    default_act = nn.SiLU()

    #g：x有输入通道数，每个卷积核，正常卷积后输出1个通道，即输入通道数运算成一个输出通道，g可以指定运算成多少个输出通道，极端g=c1时，表示每一个输入通道单独运算
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        #bias=False不需要偏置是因为后续会使用BatchNorm2d归一化
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #使用哪种激活函数，act if isinstance(act, nn.Module)：如果自己实现了激活函数，则用
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    #卷积层：卷积->归一化->激活函数
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

#Bottleneck
class Bottleneck(nn.Module):
    #add=True：是否使用残差连接，e=0.5：中间通道数缩放比例
    def __init__(self, c1, c2, k=(3,3), s=1, p=None, add=True, e=0.5):
        super().__init__()
        #中间层通道数，使用c2的原因是统一yolov8不同模型l,s,m之间的模型尺寸的缩放
        #在改动模型缩放时，通常改变配置文件的宽度因子从而改变c2的大小，Bottleneck通过c2*e保证了无论网络怎么变，内部结构的比例不变
        c = int(c2*e)
        #Bottleneck使用两层卷积，所以分别得到 k[0]和 k[1]作为核大小，(k[i]元素可能是整数，也可能是元组)
        self.conv1 = Conv(c1, c, k[0], s, p)
        #conv2 的步长应该固定为 1，下采样只由 conv1 负责
        self.conv2 = Conv(c, c2, k[1], 1, p)
        #启动残差连接，并且初始通道数等于最终通道数
        self.add = add and c1==c2

    #Bottleneck：两层卷积，是否加残差连接
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

#C3k：可变卷积核大小，扩大感受野
class C3k(nn.Module):
    #CSP是一种思想，一部分数据经过复杂计算，一部分数据不经过处理，好处是保留原始数据流，增强梯段感
    #C3是思想的落地实现
    def __init__(self, c1, c2, k=3, n=1, g=1, shortcut=True, e=0.5):
        super().__init__()
        c = int(c2*e)
        self.conv1 = Conv(c1, c, 1, 1)
        self.conv2 = Conv(c1, c, 1, 1)
        self.conv3 = Conv(2*c, c2, 1, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(c1, c2, k=(k,k), add=shortcut, e=1.0)
                for _ in range(n)
            )
        )  #n个Bottleneck，并且可以自定义卷积核大小k

    def forward(self, x):
        y1 = self.m(self.conv1(x)) #主分支，经过降维经过Bottleneck，
        y2 = self.conv2(x) #旁分支，降维后不做处理，保留原有数据特征
        y = torch.cat((y1, y2), dim=1)
        return self.conv3(y)


#C3k2，可选择使用C3k或者Bottleneck
class C3k2(nn.Module):
    def __init__(self, c1, c2, k=3, n=1, g=1, shortcut=True, e=0.5, c3k = True):
        super().__init__()
        self.c = int(c2*e)
        self.conv1 = Conv(c1, 2*self.c, 1, 1)
        self.conv2 = Conv((n+2)*self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, g=g, shortcut=shortcut, e=1.0, k=k, n=2)
            if c3k
            else Bottleneck(self.c, self.c, add=shortcut, e=1.0, k=(k,k))
            for _ in range(n)
        )  #如果c3k为true，深层可以c3k强化上下文感受野，否则使用普通Bottleneck

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, dim=1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        #多头数量，防止为0
        self.num_heads = max(num_heads, 1)
        #每个头处理的通道处
        self.head_dim = dim // num_heads
        #Q,K需降维后计算，减少计算量
        self.key_dim = int(self.head_dim * attn_ratio)
        #缩放因子，防止点积过大导致softmax饱和
        self.scale = self.head_dim ** -0.5
        #qkv总通道数,v使用原通道数
        qkv_channels = dim + 2 * self.key_dim * self.num_heads
        #使用1*1卷积生成Q,K,V
        self.qkv = Conv(dim, qkv_channels, 1, 1, act=False)
        #注意力机制后的维度投影
        self.proj = Conv(dim, dim, 1, 1, act=False)
        #位置信息,g=dim表示每个通道单独使用一个卷积
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        #空间位置数量
        N = H*W
        #生成Q,K,V特征
        qkv = self.qkv(x)
        #重排为多头格式
        qkv = qkv.view(B, self.num_heads, 2*self.key_dim+self.head_dim, N)
        #拆成q,k,v
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        #计算注意力矩阵
        attn = (q.transpose(-2, -1) @ k) * self.scale
        #softmax归一化
        attn = attn.softmax(dim=-1)
        #V加权求和
        out = v @ attn.transpose(-2, -1)
        #多头拼回所有通道
        out = out.reshape(B, C, H, W)
        #加入位置编码
        out = out + self.pe(v.reshape(B, C, H, W))
        #输出投影后的维度
        return self.proj(out)

    #疑问：1.加权求和时为什么是attn.transpose(-2, -1)。2.位置编码为什么是v.reshape(B, C, H, W)
