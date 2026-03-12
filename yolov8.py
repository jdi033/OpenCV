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
        self.conv2 = Conv(c, c2, k[1], s, p)
        #启动残差连接，并且初始通道数等于最终通道数
        self.add = add and c1==c2

    #Bottleneck：两层卷积，是否加残差连接
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

#C2f
class C2f(nn.Module):
    #n=1：使用Bottleneck的个数
    def __init__(self, c1, c2, s=1, p=None, n=1, add=True, e=0.5):
        super().__init__()
        self.c = int(c2*e)
        #第一个卷积，2*self.c：两份直接流向最后一个卷积，其中一份流向Bottleneck
        self.conv1 = Conv(c1, 2*self.c, 1, 1)
        #第二个也是最后一个卷积，(n+2)*self.c：第一个卷积的两份，n个Bottleneck的处理结果
        self.conv2 = Conv((n+2)*self.c, c2, 1)
        #for _ in range(n)：n个Bottleneck，会放在一个ModuleList集合中
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, k=((3,3),(3,3)), s=s, p=p, add=add, e=0.5) for _ in range(n))

    #C2f：卷积->n个Bottleneck->卷积
    def forward(self, x):
        #chunk(2, 1)：将第一个卷积的输出从维度1上一分为二
        y = list(self.conv1(x).chunk(2, 1))
        #y[-1]：从y集合中依次取出最后一个元素，作为Bottleneck的入参，并将结果加进y中
        y.extend(m(y[-1]) for m in self.m)
        #torch.cat(y, 1)：将y集合从维度1上合并成单个元素
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        #如果直接使用c1作为输入通道数，在经过3次最大池化层后，最后一个卷积的输入通道会是4*c1，引发输入通道和参数计算量负载
        #所以将c1//2特征减半，保留原来特征的同事，降低计算负载
        c_ = c1//2
        self.conv1 = nn.Conv2d(c1, c_, 1, 1)
        #第二个卷积，会接收第一个卷积的输出，3个MaxPool2d的输出
        self.conv2 = nn.Conv2d(c_*4, c2, 1, 1)
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
    def __init__(self, reg_max=16):
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        #生成一个长度为reg_max的数组：[0,1,2...,15]
        grid = torch.arange(reg_max, dtype=torch.float)
        #赋值self.grid形状: [1,16,1,1]
        self.register_buffer('grid', grid.view(1, reg_max, 1, 1))
        self.c1 = reg_max

    def forward(self, x):
        #b为批次大小，c为4*距离维度数(即总共需要预测的维度数)，a为该特征图的总检测点数，[10,64,6400]
        b, c, a = x.shape
        #对x调整形状: [10,64,6400]->[10,4,16,6400]->[10,16,4,6400]
        x_reshape = x.view(b, 4, self.c1, a).permute(0, 2, 1, 3)
        #对维度1使用softmax函数，使得对于每一个位置，16个距离维度的概率总和为1
        prob_dist = x_reshape.softmax(1)
        #[10,16,4,6400]->[10,1,4,6400]->[10,4,6400]
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
        #下限兜底 16 和 4*reg_max (即 64)，确保回归分支在提取特征时有足够的维度去表达 64 个概率分布；同时参考浅层特征图的宽度 ch[0]//4
        c2 = max(16, ch[0]//4, 4*self.reg_max)
        #参考浅层特征图宽度 ch[0]//4，但如果类别数 nc 极大，用 min(self.nc, 100) 来防止分类分支的中间层变得过度臃肿
        c3 = max(ch[0]//4, min(self.nc, 100))

        #分类分支，for x in ch：每个特征图，使用3个卷积，最后通道数为分类个数
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, c2, 3),
                nn.Conv2d(c2, c2, 3),
                nn.Conv2d(c2, self.nc, 1),
            ) for x in ch
        )

        #回归分支，最后通道数为检测点4个位置的距离维度
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

        #分类与回归分支，每个特征图经过卷积后的结果
        #y_cls: [[10,80,80,80],[10,80,60,60],[10,80,40,40]]
        y_cls=[]
        y_reg=[]
        for i in range(self.nl):
            #将特征图展平，cls_out:[10,80,6400],reg_out:[10,64,6400]
            cls_out = self.cv2[i](x[i])
            reg_out = self.cv3[i](x[i])
            y_cls.append(cls_out.view(shape[0], self.nc, -1))
            y_reg.append(reg_out.view(shape[0], self.reg_output_dim, -1))

        #cls_concatenated:[10, 80, 6400+1600+400], reg_concatenated:[10, 64, 6400+1600+400]
        cls_concatenated = torch.cat(y_cls, 2)
        reg_concatenated = torch.cat(y_reg, 2)


        #训练阶段
        if self.training:
            # pred:[10, 144, 6400+1600+400]
            pred = torch.cat((cls_concatenated, reg_concatenated), 1)
            return pred
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

#拼接各模块
class YOLOv8(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.nc = nc

        self.stem = Conv(3, 16, k=3, s=2)

        self.conv2 = Conv(16, 32, k=3, s=2)
        self.c2f2 = C2f(32,32,n=1,add=True)

        self.conv3 = Conv(32, 64, k=3, s=2)
        self.c2f3 = C2f(64,64,n=2,add=True)

        self.conv4 = Conv(64, 128, k=3, s=2)
        self.c2f4 = C2f(128,128,n=2,add=True)

        self.conv5 = Conv(128, 256, k=3, s=2)
        self.c2f5 = C2f(256,256,n=2,add=True)
        self.sppf = SPPF(256,256,k=5)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')