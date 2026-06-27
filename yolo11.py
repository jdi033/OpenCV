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
        #dim 必须能被 num_heads 整除
        assert dim % self.num_heads == 0
        #每个头处理的通道处
        self.head_dim = dim // self.num_heads
        #Q,K需降维后计算，减少计算量
        #降维的原因是：qk只负责寻找“谁与谁相关”，计算相关性可以不完全计算所有维度，而且v保留全部信息，使用的是原始维度
        #对于信息缺失，yolo11还做了残差连接（保留原始信息），PE补全空间位置信息，FFN重新做通道融合
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
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
        self.conv1 = Conv(c1, c_, 1, 1, act=False)
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


class YOLO11(nn.Module):
    def __init__(self, nc=2):
        super().__init__()

        self.nc = nc

        #关于是否使用C3K模块，浅层负责小划痕、边缘、微小缺陷，不宜过度扩大感受野；深层负责大面积污渍、上下文语义，更适合 C3k
        #Backbone
        self.stem = Conv(3, 16, 3, 2)
        self.conv2 = Conv(16, 32, 3, 2)
        self.c3k2_2 = C3k2(32, 32, n=1, shortcut=True, c3k=False)

        #backbone_p3
        self.conv3 =  Conv(32, 64, 3, 2)
        self.c3k2_3 = C3k2(64, 64, n=2, shortcut=True, c3k=False)

        #backbone_p4
        self.conv4 = Conv(64, 128, 3, 2)
        self.c3k2_4 = C3k2(128, 128, n=2, shortcut=True, c3k=True)

        #backbone_p5
        self.conv5 = Conv(128, 256, 3, 2)
        self.c3k2_5 = C3k2(256, 256, n=1, shortcut=True, c3k=True)
        self.sppf = SPPF(256, 256, k=5)
        self.c2psa = C2PSA(256, 256, n=2)

        #Neck
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_c3k2_1 = C3k2(128+256, 128, n=1, shortcut=False, c3k=False)

        #neck_p3
        self.neck_c3k2_2 = C3k2(64+128, 64, n=1, shortcut=False, c3k=False)

        #neck_p4
        self.neck_conv1 = Conv(64, 64, 3, 2)
        self.neck_c3k2_3 = C3k2(64+128, 128, n=1, shortcut=False, c3k=False)

        # neck_p5
        self.neck_conv2 = Conv(128, 128, 3, 2)
        self.neck_c3k2_4 = C3k2(128+256, 256, n=1, shortcut=False, c3k=True)

        #Detect
        self.head = Detect(self.nc, ch=(64, 128, 256))

    def forward(self, x):
        #Backbone
        x = self.stem(x)
        p2 = self.c3k2_2(self.conv2(x))
        p3 = self.c3k2_3(self.conv3(p2))
        p4 = self.c3k2_4(self.conv4(p3))
        p5 = self.c2psa(self.sppf(self.c3k2_5(self.conv5(p4))))

        #Neck
        up_p5 = self.upsample(p5)
        out_neck1 = self.neck_c3k2_1(torch.cat([p4, up_p5], 1))
        up_neck1 = self.upsample(out_neck1)
        out_p3 = self.neck_c3k2_2(torch.cat([p3, up_neck1], 1))

        down_p3 = self.neck_conv1(out_p3)
        out_p4 = self.neck_c3k2_3(torch.cat([down_p3, out_neck1], 1))

        down_p4 = self.neck_conv2(out_p4)
        out_p5 = self.neck_c3k2_4(torch.cat([down_p4, p5], 1))

        #Detect
        pred = self.head([out_p3, out_p4, out_p5])

        return pred

#计算交并比
def bbox_iou(box1, box2):
    #box1预测框[B,N,4] box2真实框[B,M,4]

    #增加维度，方便计算
    b1 = box1.unsqueeze(1)
    b2 = box2.unsqueeze(2)
    #计算交集的左上角和右下角
    inter_l = torch.max(b1[..., :2], b2[..., :2])
    inter_r = torch.min(b1[..., 2:], b2[..., 2:])
    #交集的高宽
    inter_wh = (inter_r - inter_l).clamp(min=0)
    #交集面积
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    #分别计算两个框的面积
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    #并集面积
    union_area = area1 + area2 - inter_area + 1e-16
    #返回交并比IoU
    return inter_area / union_area

#任务对齐分配器
class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        super().__init__()
        #为每个真实缺陷的候选正样本数量
        self.topk = topk
        #分类得分的权重
        self.alpha = alpha
        #交并比的权重
        self.beta = beta

    def forward(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes):
        #pred_scores: 预测的分类得分[B,N,nc]  pred_bboxes: 预测的坐标[B,N,4]
        #gt_labels: 真实的分类标签[B,M,1]  gt_bboxes: 真实的缺陷框坐标[B,M,4]
        B, M, _ = gt_bboxes.shape
        B, N, nc = pred_scores.shape
        #扩展维度[B,M,1]->[B,M,N]，以便gather操作
        # 核心修复：填充框防御机制
        # 1. 生成真实框的掩码 (把为了对齐 Batch 而填充的 -1 剔除)
        # 如果标签 > -1，就是真实的缺陷，mask 值为 1.0；否则为 0.0
        mask_gt = (gt_labels > -1.0).float().expand(B, M, N)  # [B, M, N]

        # 2. 将 最小值-1 改成 0，防止底层 CUDA 的 gather 操作数组越界崩溃！
        # 因为有 mask_gt 兜底，这里借用 0 不会产生任何实质影响
        target_labels = gt_labels.long().clamp(min=0).expand(B, M, N)
        #预测分类对于真是分类的分类得分
        #scores:[B,M,N], 每个缺陷对应每个预测点在该类别的预测概率
        scores = pred_scores.permute(0, 2, 1).gather(1, target_labels)
        #交并比，ious:[B,M,N]
        ious = bbox_iou(pred_bboxes, gt_bboxes)
        ious = ious.clamp(min=0)
        #计算对交分数，某个预测点对应某个缺陷的匹配度[B,M,N]
        # 核心修复：抹杀假框得分
        # 计算对齐分数时，必须乘上 mask_gt，把那些假框的得分直接清零！
        alignment_metrics = (scores ** self.alpha) * (ious ** self.beta) * mask_gt
        #取前topk个分类最高得分[B,M,topk]
        topk_metrics, topk_idxs = torch.topk(alignment_metrics, self.topk, dim=-1)

        #构建掩码
        mask_pos = torch.zeros_like(alignment_metrics)
        #将topk_ids位置赋值1
        mask_pos = mask_pos.scatter_(-1, topk_idxs, 1.0)
        # 过滤虽然进了topk，但得分低 (这里的 1e-9 也能顺便把全 0 的假框彻底过滤掉)
        mask_pos = mask_pos * (alignment_metrics > 1e-9).float()

        #独立原则，如果一个预测点同时入选多个缺陷的topk，只去分数最高的,[B,N]
        max_metrics, max_idxs = alignment_metrics.max(dim=1)
        #最终的正样本mask[B,M,N]
        is_max_mask = (alignment_metrics == max_metrics.unsqueeze(1)).float()

        #即入选topk，又满足独立原则
        mask_pos = mask_pos * is_max_mask

        return mask_pos, alignment_metrics

def bbox_iou_1v1(box1, box2):
    #box1: [Number_pos, 4]
    inter_l = torch.max(box1[:, :2], box2[:, :2])
    inter_r = torch.min(box1[:, 2:], box2[:, 2:])

    inter_wh = (inter_r - inter_l).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = area1 + area2 - inter_area + 1e-16

    return inter_area / union_area

def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1, b1_x2, b1_y1, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_x2, b2_y1, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #计算交集
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    #计算并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    #基础iou
    iou = inter_area / union_area

    #两个框的最小包围框，即把两个框全部包含在内
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    #中心点距离平方
    pred_cx, pred_cy = (b1_x1+b1_x2)/2, (b1_y1+b1_y2)/2
    gt_cx, gt_cy = (b2_x1+b2_x2)/2, (b2_y1+b2_y2)/2
    rho2 = (pred_cx-gt_cx)**2 + (pred_cy-gt_cy)**2

    # CIoU 额外项：宽高比一致性惩罚 v
    # arctan(w/h) 的差异，衡量宽高比一致性
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )

    # 动态权重 alpha
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    #最终iou
    ciou = iou - (rho2 / c2 + v * alpha)

    return ciou

class v8DetectionLoss(nn.Module):
    def __init__(self, nc=2, reg_max=16):
        super(v8DetectionLoss, self).__init__()
        self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_max = reg_max
        #将DFL模块放进Loss中
        self.dfl = DFL(reg_max) if reg_max > 0 else nn.Identity()
        #特征图步长，先在这里写死
        self.strides = [8,16,32]

    def forward(self, pred_scores, pred_dist, gt_labels, gt_bboxes, image_shape=(640,640)):
        #pred_dist为回归分支处理的最原始64维分布特征[B,N,64]
        B,N,nc = pred_scores.shape

        device = pred_scores.device

        #特征图尺寸
        feats_shape = [(image_shape[0]//s, image_shape[1]//s) for s in self.strides]
        #锚点的位置[N,2]
        # 接收两个返回值
        anchor_points, stride_tensor = make_anchor(feats_shape, self.strides)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)  # 🌟 步长也送入 GPU
        #anchor_points = make_anchor(feats_shape, self.strides).to(pred_scores.device)
        #pred_dist为[B,N,64],而DFL需要[B,64,N]
        dist_permuted = pred_dist.permute(0,2,1)
        #pred_ltrb:相对距离[B,4,N]
        pred_ltrb = self.dfl(dist_permuted)
        #再次转置[B,N,4]
        pred_ltrb = pred_ltrb.permute(0,2,1)
        #网格距离 × 步长 = 真实像素距离！
        pred_ltrb = pred_ltrb * stride_tensor
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=False, dim=-1)

        #no_grad告诉pytorch，以下不参数梯度计算。.detach()会生成一个新的张量，并且计算不会返回修改权重
        with torch.no_grad():
            mask_pos, alignment_metrics = self.assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes.detach(),
                gt_labels,
                gt_bboxes)
            #找到所有正样本，以及他们负责的缺陷索引，fg_mask, target_gt_idx:[B,N]
            fg_mask, target_gt_idx = mask_pos.max(dim=1)
            #正样本个数,torch.clamp 确保分母至少为 1.0，防止除零
            target_scores_sum = torch.clamp(fg_mask.sum(), min=1.0)

        #计算分类损失
        #初始化分类目标矩阵[B,N,nc]
        target_scores = torch.zeros_like(pred_scores)
        #每个预测点在所有缺陷上的最高分类得分
        max_metrics, _ = alignment_metrics.max(dim=1)

        for b in range(B):
            #在b张图片上，不为0的正样本的位置下标索引
            pos_idxs = fg_mask[b].nonzero().squeeze(-1)
            #防止正样本个数小于0
            if pos_idxs.numel() > 0:
                #正样本预测的是哪个缺陷
                assigned_gt_idx = target_gt_idx[b, pos_idxs]
                #这些缺陷对用哪个类别
                pos_target_lables = gt_labels[b,assigned_gt_idx,0].long()
                #正样本的最高得分作为计算损失函数的目标分类得分
                target_scores[b,pos_idxs,pos_target_lables] = max_metrics[b, pos_idxs]

        #pred_scores预测分类得分，target_scores该预测点应该预测的缺陷以及最高得分，最后target_scores_sum总正样本数算平均
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        #计算回归损失 (CIoU) 与分布损失 (DFL) 协同结算
        #zeros(1)创建一维的元素为0的张量，
        #device=pred_scores.device使得创建的loss_box张量和网络输出的张量在同一张显卡上，不会设备报错
        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)

        if fg_mask.sum() > 0:
            for b in range(B):
                pos_idxs = fg_mask[b].nonzero().squeeze(-1)
                #这些正样本预测的回归框
                pos_pred_bboxes = pred_bboxes[b, pos_idxs]
                #正样本对应的缺陷
                assigned_gt_idx = target_gt_idx[b, pos_idxs]
                #这些缺陷的真实框
                pos_gt_bboxes = gt_bboxes[b, assigned_gt_idx]
                #计算预测框与真实框的iou
                bbox_iou = bbox_ciou(pos_pred_bboxes, pos_gt_bboxes)

                #计算总回归损失，iou越大越好，使用(1-iou)作为惩罚
                loss_box += (1-bbox_iou).sum()

                # DFL 损失计算 (为 64 维分布灌入监督信号)
                pos_anchors = anchor_points[pos_idxs]
                pos_strides = stride_tensor[pos_idxs]

                # 逆向推导真实边界框到正样本锚点中心的真实网格距离 (gt_ltrb)
                gt_lt = pos_anchors - pos_gt_bboxes[:, :2]
                gt_rb = pos_gt_bboxes[:, 2:] - pos_anchors
                # 除以对应特征层的步长，转换到真实的网格坐标系尺度
                gt_ltrb = torch.cat([gt_lt, gt_rb], dim=-1) / pos_strides
                # 边界防御：限制在 0 到 reg_max-1 之间，防止越界
                gt_ltrb = gt_ltrb.clamp(min=0, max=self.reg_max - 1.0 - 1e-4)

                # 提取正样本对应的原始 64 维预测分布并对齐维度
                pos_pred_dist = pred_dist[b, pos_idxs]  # [Num_pos, 64]
                pos_pred_dist = pos_pred_dist.view(-1, 4, self.reg_max).view(-1, self.reg_max)  # [Num_pos * 4, 16]
                flat_gt_ltrb = gt_ltrb.view(-1)  # [Num_pos * 4]

                # 计算 DFL 的双侧分数交叉熵
                tl = flat_gt_ltrb.long()
                tr = tl + 1
                wl = tr.float() - flat_gt_ltrb
                wr = 1.0 - wl

                loss_dfl_step = (
                        F.cross_entropy(pos_pred_dist, tl, reduction='none') * wl +
                        F.cross_entropy(pos_pred_dist, tr, reduction='none') * wr
                )
                loss_dfl += loss_dfl_step.sum()

            # 统一按正样本总数做均值归一化
            loss_box = loss_box / fg_mask.sum()
            loss_dfl = loss_dfl / fg_mask.sum()

        #超参数权重协同配比 (7.5 与 1.5)
        total_loss = loss_cls * 1.0 + loss_box * 7.5 + loss_dfl * 1.5
        return total_loss

#预测框对锚点的相对距离 + 锚点位置 -> 绝对距离
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    #distance[B,N,4](l,t,r,b), anchor_points[N,2](cx,cy)
    #lt[B,N,2]
    lt, rb = distance.chunk(2, dim)
    #左上，右下坐标
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1+x2y2)/2
        wh = x2y2-x1y1
        #返回格式为：[B,N,4](cx,cy,w,h)
        return torch.cat((c_xy, wh), dim=dim)

    # 返回格式为：[B,N,4](x1,y1,x2,y2)
    return torch.cat((x1y1, x2y2), dim=dim)


#根据特征图尺寸和步长，将8400个坐标计算出锚点坐标
def make_anchor(feats_shape, strides, grid_cell_offset=0.5):
    #feats_shape特征图尺寸[P3,P4,P5][(80,80),(40,40),(20,20)],strides步长[8,16,32],grid_cell_offset网格中心偏移量
    anchor_points = []
    # 记录每个点对应的步长
    stride_tensor = []
    for i, stride in enumerate(strides):
        h,w = feats_shape[i]
        #小网格的位置索引,[h,w]
        stride_y, stride_x = torch.meshgrid(torch.arange(end=h), torch.arange(end=w), indexing='ij')
        #torch.stack((stride_y, stride_x), dim=-1):[h,w,2],view(-1, 2)展平为[h*w,2]
        anchors = torch.stack((stride_x, stride_y), dim=-1).view(-1, 2) + grid_cell_offset
        anchor_points.append(anchors*stride)

        # 生成对应数量的步长并记录 [h*w, 1]
        stride_tensor.append(torch.full((h * w, 1), stride))
    #P3, P4, P5 的锚点在第 0 维度拼接起来: 6400 + 1600 + 400 = 8400,[8400, 2]
    # 返回两个值：绝对锚点坐标，以及对应的步长张量
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)