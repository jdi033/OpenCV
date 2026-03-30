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
            #pred = torch.cat((reg_concatenated, cls_concatenated), 1)
            #return pred
            # 不要 torch.cat 拼接了，直接返回分离的特征图！
            # cls_concatenated: [B, 80, 8400] -> 转置为 [B, 8400, 80] (适应 Loss 接收)
            # reg_concatenated: [B, 64, 8400] -> 转置为 [B, 8400, 64]
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

        self.c2f_neck1 = C2f(256+128,128,n=1,add=False)

        self.c2f_neck2 = C2f(128+64,64,n=1,add=False)

        self.conv_neck1 = Conv(64, 64, k=3, s=2)

        self.c2f_neck3 = C2f(128+64,128,n=1,add=False)

        self.conv_neck2 = Conv(128, 128, k=3, s=2)

        self.c2f_neck4 = C2f(256+128,256,n=1,add=False)

        self.head = Detect(self.nc, ch=(64,128,256))

    def forward(self, x):
        x = self.stem(x)
        x = self.c2f2(self.conv2(x))
        p3 = self.c2f3(self.conv3(x))
        p4 = self.c2f4(self.conv4(p3))
        p5 = self.sppf(self.c2f5(self.conv5(p4)))

        up_p5 = self.upsample(p5)
        out_neck1 = self.c2f_neck1(torch.cat([up_p5, p4], 1))
        up_neck1 = self.upsample(out_neck1)
        out_p3 = self.c2f_neck2(torch.cat([up_neck1, p3], 1))
        down_p3 = self.conv_neck1(out_p3)
        out_p4 = self.c2f_neck3(torch.cat([out_neck1, down_p3], 1))
        down_p4 = self.conv_neck2(out_p4)
        out_p5 = self.c2f_neck4(torch.cat([p5, down_p4], 1))

        pred = self.head([out_p3, out_p4, out_p5])

        return pred


# if __name__ == '__main__':
#     model = YOLOv8(nc=2)
#     model.train()
#     dummy_image = torch.randn(1, 3, 640, 640)
#     out = model(dummy_image)
#
#     if out.shape == (1,66,8400):
#         print("success, out shape is {}".format(out.shape))
#     else:
#         print("fail, out shape is {}".format(out.shape))

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
        target_labels = gt_labels.long().expand(B,M,N)
        #预测分类对于真是分类的分类得分
        #scores:[B,M,N], 每个缺陷对应每个预测点在该类别的预测概率
        scores = pred_scores.permute(0, 2, 1).gather(1, target_labels)
        #交并比，ious:[B,M,N]
        ious = bbox_iou(pred_bboxes, gt_bboxes)
        ious = ious.clamp(min=0)
        #计算对交分数，某个预测点对应某个缺陷的匹配度[B,M,N]
        alignment_metrics = (scores**self.alpha) * (ious**self.beta)
        #取前topk个分类最高得分[B,M,topk]
        topk_metrics, topk_idxs = torch.topk(alignment_metrics, self.topk, dim=-1)

        #构建掩码
        mask_pos = torch.zeros_like(alignment_metrics)
        #将topk_ids位置赋值1
        mask_pos = mask_pos.scatter_(-1, topk_idxs, 1.0)
        #过滤虽然进了topk，但得分低
        mask_pos = mask_pos*(alignment_metrics>1e-9).float()

        #独立原则，如果一个预测点同时入选多个缺陷的topk，只去分数最高的,[B,N]
        max_metrics, max_idxs = alignment_metrics.max(dim=1)
        #最终的正样本mask[B,M,N]
        is_max_mask = (alignment_metrics == max_metrics.unsqueeze(1)).float()

        #即入选topk，又满足独立原则
        mask_pos = mask_pos * is_max_mask

        return mask_pos, alignment_metrics

# if __name__ == '__main__':
#     assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
#
#     #伪造预测数据
#     B,N,nc = 1, 8400,2
#     pred_scores = torch.randn(B, N, nc).sigmoid()
#     pred_bboxes = torch.randn(B, N, 4)*640
#     pred_bboxes[..., 2:] += pred_bboxes[..., :2]
#
#     #伪造真实缺陷数据
#     M=1
#     gt_labels = torch.tensor([[[1.0]]])
#     gt_bboxes = torch.tensor([[[100.0, 100.0, 120.0, 120.0]]])
#
#     mask_pos, alignment_metrics = assigner(pred_scores, pred_bboxes, gt_labels, gt_bboxes)
#
#     print(f"生成的正样本掩码 (Mask) 维度: {mask_pos.shape}")
#
#     # 统计有多少个点被选为了正样本去负责预测这个“砂眼”
#     num_positives = mask_pos.sum().item()
#     print(f"🎯 在 8400 个预测点中，有 {int(num_positives)} 个点被提拔为 '正样本'。")
#     print(f"👻 剩下的 {8400 - int(num_positives)} 个点将被无情压制，计算背景 Loss。")



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

        #特征图尺寸
        feats_shape = [(image_shape[0]//s, image_shape[1]//s) for s in self.strides]
        #锚点的位置[N,2]
        anchor_points = make_anchor(feats_shape, self.strides).to(pred_scores.device)
        #pred_dist为[B,N,64],而DFL需要[B,64,N]
        dist_permuted = pred_dist.permute(0,2,1)
        #pred_ltrb:相对距离[B,4,N]
        pred_ltrb = self.dfl(dist_permuted)
        #再次转置[B,N,4]
        pred_ltrb = pred_ltrb.permute(0,2,1)
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

        #计算回归损失
        #zeros(1)创建一维的元素为0的张量，
        #device=pred_scores.device使得创建的loss_box张量和网络输出的张量在同一张显卡上，不会设备报错
        loss_box = torch.zeros(1, device=pred_scores.device)

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
                bbox_iou = bbox_iou_1v1(pos_pred_bboxes, pos_gt_bboxes)

                #计算总回归损失，iou越大越好，使用(1-iou)作为惩罚
                loss_box += (1-bbox_iou).sum()

            #平均值
            loss_box = loss_box / fg_mask.sum()

        #1.5为权重放大系数
        return loss_cls + loss_box*1.5


if __name__ == "__main__":
    print("🚀 启动网络完整闭环：前向计算 + 损失算账 + 梯度反传...")

    # 1. 模拟网络前向传播产生的预测结果 (要设置 requires_grad=True 让张量带有梯度属性)
    B, N, nc = 1, 8400, 2
    # 注意：这里模拟的是 Detect 头没经过 sigmoid 的原始输出 Logits
    dummy_pred_scores = torch.randn(B, N, nc, requires_grad=True)
    dummy_pred_bboxes = (torch.rand(B, N, 4) * 640).clone().detach().requires_grad_(True)

    # 2. 模拟真实标签 (GT)
    gt_labels = torch.tensor([[[1.0]]])
    gt_bboxes = torch.tensor([[[100.0, 100.0, 120.0, 120.0]]])

    # 3. 实例化我们刚写的结算中心
    criterion = v8DetectionLoss(nc=2)

    # 4. === 前向算账 (Forward Pass) ===
    total_loss = criterion(dummy_pred_scores, dummy_pred_bboxes, gt_labels, gt_bboxes)

    print(f"💰 网络在当前随机状态下，对这张图像评估的‘糟糕程度’ (Total Loss): {total_loss.item():.4f}")

    # 5. === 见证奇迹的时刻：反向求导 (Backward Pass) ===
    # 这行代码会沿着计算图一路往回走，自动算出每一个卷积核为了降低这个 Loss 应该如何修改
    total_loss.backward()

    # 6. 验证梯度是否真的传回来了
    print(f"⚡ 预测分类张量是否获得了梯度 (Gradient)？: {dummy_pred_scores.grad is not None}")

    if dummy_pred_scores.grad is not None:
        # 打印出前 5 个网格点的第一个类别的梯度数值看看
        print(f"📊 前 5 个预测点的梯度值摘录:\n{dummy_pred_scores.grad[0, :5, 0]}")
        print("\n🎉 恭喜你！从主干网络到 Detect 头，再到 Assigner 和 Loss，你手工打造的深度学习引擎已经可以完美运转了！")


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
    #
    anchor_points = []
    for i, stride in enumerate(strides):
        h,w = feats_shape[i]
        #小网格的位置索引,[h,w]
        stride_y, stride_x = torch.meshgrid(torch.arange(end=h), torch.arange(end=w), indexing='ij')
        #torch.stack((stride_y, stride_x), dim=-1):[h,w,2],view(-1, 2)展平为[h*w,2]
        anchors = torch.stack((stride_x, stride_y), dim=-1).view(-1, 2) + grid_cell_offset
        anchor_points.append(anchors*stride)

    #P3, P4, P5 的锚点在第 0 维度拼接起来: 6400 + 1600 + 400 = 8400,[8400, 2]
    return torch.cat(anchor_points, dim=0)