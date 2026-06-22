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
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            dilation=d,
            groups=g,
            bias=False
            )
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
                Bottleneck(c, c, k=(k,k), add=shortcut, e=1.0)
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
        self.head_dim = dim // self.num_heads
        #Q,K需降维后计算，减少计算量
        #降维的原因是：qk只负责寻找“谁与谁相关”，计算相关性可以不完全计算所有维度，而且v保留全部信息，使用的是原始维度
        #对于信息缺失，yolo11还做了残差连接（保留原始信息），PE补全空间位置信息，FFN重新做通道融合
        self.key_dim = int(self.head_dim * attn_ratio)
        #缩放因子，防止点积过大导致softmax饱和
        self.scale = self.key_dim ** -0.5
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
        #attn[B, heads, N, N]，后两位attn[i,j]表示第i个位置对第j个位置的关注权重，而v[B,heads,head_dim,N]，后一位v[j]表示位置j的信息
        #我们想要的是：位置i上，所有位置j的加权汇总信息，即out[:, i] = Σ_j A[i, j] * V[:, j](最后一位j上数据相乘)
        #而v的最后两位是[head_dim, j]，所以只能点积attn_T[j,i]，最后的结果是[head_dim,i]，最后一位就表示位置i的所有信息加权
        out = v @ attn.transpose(-2, -1)
        #多头拼回所有通道
        out = out.reshape(B, C, H, W)
        #加入位置编码，使用v的原因：v表示该位置的内容信息，v是经过qkv投影后的特征，out与v是同一特征空间，再使用3*3提取局部位置关系
        out = out + self.pe(v.reshape(B, C, H, W))
        #输出投影后的维度
        return self.proj(out)

#PSA 基础残差块
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.add = shortcut
        #注意力负责空间位置之间的信息交互
        self.attn = Attention(c, num_heads=max(num_heads, 1), attn_ratio=attn_ratio)
        #通道维度上的非线性变换，FFN负责通道之间的信息重组
        self.ffn = nn.Sequential(
            #使用1*1卷积，增强通道交互
            Conv(c, c*2, 1, 1),
            #再使用1*1卷积，降回原通道
            Conv(c*2, c, 1, 1, act=False),
        )

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

#深层注意力模块
class C2PSA(nn.Module):
    #C2分流结构 + PSA 注意力，用于深层语义增强
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        #C2PSA要求输入通道数等于输出通道数
        assert c1==c2
        #隐藏层通道数
        self.c = int(c1*e)
        #使用1*1卷积拆成2份
        self.conv1 = Conv(c1, 2*self.c, 1, 1)
        #将2份拼回原有通道
        self.conv2 = Conv(2*self.c, c2, 1, 1)
        #n个C2PSA模块
        self.m = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c//64, 1))
                for _ in range(n)
            )
        )

    def forward(self, x):
        a, b = self.conv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.conv2(torch.cat((a, b), dim=1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        #如果直接使用c1作为输入通道数，在经过3次最大池化层后，最后一个卷积的输入通道会是4*c1，引发输入通道和参数计算量负载
        #所以将c1//2特征减半，保留原来特征的同事，降低计算负载
        c_ = c1//2
        self.conv1 = Conv(c1, c_, 1, 1)
        #第二个卷积，会接收第一个卷积的输出，3个MaxPool2d的输出
        self.conv2 = Conv(c_*4, c2, 1, 1)
        #最大池化层，保证输入与输出通道数不变
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    #SPPF：卷积->3个最大池化层->卷积
    def forward(self, x):
        y = self.conv1(x)
        y1 = self.m(y)
        y2 = self.m(y1)
        return self.conv2(torch.cat((y, y1, y2, self.m(y2)), 1))

#DFL
class DFL(nn.Module):
    # DFL：回归预测时，预测框距离监测点的位置(l,d,r,u)的概率预测
    #reg_max=16：距离检测点每个位置的距离维度数
    # DFL不直接预测一个距离值，而是预测一个概率分布。每个方向（左、上、右、下）各预测 reg_max=16 个离散距离档位（0, 1, 2,..., 15）的概率，最后加权求和得到最终距离。
    #每个锚点的回归预测 = 4个方向 × 16个距离档位 = 64维
    def __init__(self, reg_max=16):
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        #生成一个长度为reg_max的数组：[0,1,2...,15]
        grid = torch.arange(reg_max, dtype=torch.float)
        #赋值self.grid形状: [1,16,1,1]
        #self.register_buffer('grid', grid.view(1, reg_max, 1, 1))
        self.conv.weight.data[:] = nn.Parameter(grid.view(1, reg_max, 1, 1))
        self.c1 = reg_max

    def forward(self, x):
        #b为批次大小，c为4*距离维度数(即总共需要预测的维度数)，a为该特征图的总检测点数，[10,64,6400]
        b, c, a = x.shape
        #对x调整形状: [10,64,6400]->[10,4,16,6400]->[10,16,4,6400]
        x_reshape = x.view(b, 4, self.c1, a).permute(0, 2, 1, 3)
        #对维度1使用softmax函数，使得对于每一个位置，16个距离维度的概率总和为1
        prob_dist = x_reshape.softmax(1)
        #[10,16,4,6400]->[10,1,4,6400]->[10,4,6400]
        #[B, 64, N] → [B, 4, 16, N](16个档位的概率) → softmax → 加权求和(得到l, t, r, b四个距离值) → [B, 4, N]
        return self.conv(prob_dist).view(b, 4, a)

class Detect(nn.Module):
    dynamic = False
    export = False
    #nc=80:最终的分类个数，ch=()：特征图(P3,P4,P5)
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        #回归预测的每个检测点的维度(位置个数*距离维度)
        self.reg_output_dim = 4*self.reg_max
        #分类+回归的总维度数
        self.no = nc + 4*self.reg_max
        self.stride = torch.zeros(len(ch))

        #分类和回归中间通道数
        # 参考浅层特征图宽度 ch[0]//4，但如果类别数 nc 极大，用 min(self.nc, 100) 来防止分类分支的中间层变得过度臃肿
        c2 = max(ch[0] // 4, min(self.nc, 100))
        #下限兜底 16 和 4*reg_max (即 64)，确保回归分支在提取特征时有足够的维度去表达 64 个概率分布；同时参考浅层特征图的宽度 ch[0]//4
        c3 = max(16, ch[0]//4, 4*self.reg_max)

        #分类分支，for x in ch：每个特征图，使用3个卷积，最后通道数为分类个数
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, self.nc, 1),
            ) for x in ch
        )

        #回归分支，最后通道数为检测点4个位置的距离维度
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, 4*self.reg_max, 1),
            ) for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape

        #分类与回归分支，每个特征图经过卷积后的结果
        #y_cls: [[10,80,80,80],[10,80,40,40],[10,80,20,20]]
        y_cls=[]
        y_reg=[]
        for i in range(self.nl):
            # cls_out: [B, nc, H, W]([B, 80, 80, 80]), reg_out: [B, 64, H, W]([B, 64, 80, 80])
            #空间维度 H×W 就是该层的锚点数量（如 80×80=6400）
            cls_out = self.cv2[i](x[i])
            reg_out = self.cv3[i](x[i])
            #view后：[B, 80, 6400]（6400个锚点，每个锚点在每种类别的得分）。 [B, 64, 6400]（每个锚点在每个维度的回归）
            y_cls.append(cls_out.view(shape[0], self.nc, -1))
            y_reg.append(reg_out.view(shape[0], self.reg_output_dim, -1))

        #cls_concatenated:[10, 80, 6400+1600+400], reg_concatenated:[10, 64, 6400+1600+400]
        cls_concatenated = torch.cat(y_cls, 2)
        reg_concatenated = torch.cat(y_reg, 2)


        #训练阶段
        if self.training:
            #训练时 — 返回两个张量，供 Loss 函数分别处理
            # cls_concatenated: [B, 80, 8400] -> 转置为 [B, 8400, 80] (BCEWithLogitsLoss 计算分类损失)
            # reg_concatenated: [B, 64, 8400] -> 转置为 [B, 8400, 64] (DFL 解码后计算 IoU 回归损失)
            return cls_concatenated.permute(0, 2, 1), reg_concatenated.permute(0, 2, 1)
        #推理阶段
        else:
            #在yolo中，多类别分类不能使用softmax，因为softmax只能用于类别之间不冲突(概率之和为1)，yolo可以预测出多个类别，是将多分类独立成多个二分类任务
            #使用 Sigmoid 激活函数，将每个独立通道的 Logit 压缩到[0,1]之间
            cls_scores = cls_concatenated.sigmoid()
            #得到4个位置的距离，[10,4,6400+1600+400]
            reg_results = self.dfl(reg_concatenated)
            #[10,84,6400+1600+400]
            out = torch.cat((cls_scores, reg_results), 1)
            return out