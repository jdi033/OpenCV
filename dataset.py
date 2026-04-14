import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    """
    极简版 YOLO 数据集加载器
    核心功能：读取图片与标签 -> Letterbox等比缩放 -> 坐标转换 -> 张量对齐输出
    """

    def __init__(self, img_dir, label_dir, img_size=640):
        """
        初始化数据集。
        Args:
            img_dir: 图片文件夹路径
            label_dir: YOLO格式的 .txt 标签文件夹路径
            img_size: 网络要求的输入尺寸，例如 640
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # 获取所有图片的绝对路径列表
        # 假设图片格式为 .jpg
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        # 返回数据集的总样本数
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        核心方法：当 DataLoader 迭代时，每次调用此方法获取一个样本。
        """
        # ==========================================
        # 步骤 1: 读取图片
        # ==========================================
        img_path = self.img_files[idx]
        # cv2.imread 读取的图片格式为 BGR，维度为 [H, W, C] (例如: [1080, 1920, 3])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"图片读取失败: {img_path}")

        # 将 BGR 转换为 RGB (PyTorch 模型通常使用 RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 记录原始尺寸，用于后续计算坐标缩放比例
        h_orig, w_orig = img.shape[:2]

        # ==========================================
        # 步骤 2: 读取 YOLO 格式的标签
        # YOLO 格式: [class_id, cx, cy, w, h] (全部是 0~1 的归一化比例)
        # ==========================================
        # 替换扩展名获取对应的 txt 标签路径
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))

        # labels 用于存放当前图片的所有真实缺陷数据
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # 将 txt 中的一行字符串按空格切分，转为 float 列表
                    # l 格式: [cls_id, cx, cy, w, h]
                    l = np.array(line.strip().split(), dtype=np.float32)
                    labels.append(l)

        # 如果 labels 不为空，将其转为 numpy 矩阵；否则给一个形状为 [0, 5] 的空矩阵
        # labels 维度: [M, 5]，其中 M 是缺陷数量
        labels = np.array(labels) if len(labels) > 0 else np.zeros((0, 5), dtype=np.float32)

        # ==========================================
        # 步骤 3: Letterbox 等比例缩放与灰边填充 (核心几何逻辑)
        # ==========================================
        # 3.1 计算缩放比例 (以长边为准，防止超出 640)
        r = min(self.img_size / h_orig, self.img_size / w_orig)

        # 3.2 计算等比缩放后的新宽高 (此时不带灰边)
        new_unpad_w = int(round(w_orig * r))
        new_unpad_h = int(round(h_orig * r))

        # 3.3 对原图进行缩放 (cv2.resize 输入格式为 W, H)
        # img 维度变化: [h_orig, w_orig, 3] -> [new_unpad_h, new_unpad_w, 3]
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # 3.4 计算需要填充的灰边大小 (总像素差除以 2，使得图像居中)
        dw = (self.img_size - new_unpad_w) / 2
        dh = (self.img_size - new_unpad_h) / 2

        # 为了使用 cv2.copyMakeBorder，需要将小数转为整数的上下左右填充量
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 3.5 给图片加上灰边 (默认灰色 RGB=(114, 114, 114))
        # 最终 img 维度: [640, 640, 3]
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        # ==========================================
        # 步骤 4: 坐标还原与映射 (从归一化 cxcywh 到绝对 x1y1x2y2)
        # ==========================================
        # 工业实战注: Loss 函数需要的是在 640x640 填充后图像上的绝对角点坐标！
        gt_bboxes = np.zeros((len(labels), 4), dtype=np.float32)  # [M, 4]
        gt_labels = np.zeros((len(labels), 1), dtype=np.float32)  # [M, 1]

        if len(labels) > 0:
            # 取出归一化的 cx, cy, w, h
            cx = labels[:, 1]
            cy = labels[:, 2]
            w = labels[:, 3]
            h = labels[:, 4]

            # 4.1 将归一化坐标还原为原图的绝对物理坐标，然后再乘以缩放比例 r，最后加上填充的偏移量 dw/dh
            # 这样就完美映射到了 640x640 的那张新图上
            cx_pad = cx * w_orig * r + dw
            cy_pad = cy * h_orig * r + dh
            w_pad = w * w_orig * r
            h_pad = h * h_orig * r

            # 4.2 将 cx, cy, w, h 转换为 x1, y1, x2, y2 (左上角和右下角)
            gt_bboxes[:, 0] = cx_pad - w_pad / 2  # x1
            gt_bboxes[:, 1] = cy_pad - h_pad / 2  # y1
            gt_bboxes[:, 2] = cx_pad + w_pad / 2  # x2
            gt_bboxes[:, 3] = cy_pad + h_pad / 2  # y2

            # 提取类别 ID
            gt_labels[:, 0] = labels[:, 0]

        # ==========================================
        # 步骤 5: 转换为网络认识的 PyTorch Tensors
        # ==========================================
        # 图片归一化：将像素值从 0~255 压缩到 0.0~1.0
        img_tensor = img_padded.astype(np.float32) / 255.0

        # 维度转换: OpenCV 读取的是 [H, W, C]，PyTorch 卷积层需要的是 [C, H, W]
        # [640, 640, 3] -> [3, 640, 640]
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)

        # 标签和坐标转为 Tensor
        gt_labels_tensor = torch.from_numpy(gt_labels)  # [M, 1]
        gt_bboxes_tensor = torch.from_numpy(gt_bboxes)  # [M, 4]

        return img_tensor, gt_labels_tensor, gt_bboxes_tensor