# train.py (新建的总控文件)
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# ==========================================
# 工业级模块化引入：从你写的另两个文件导入核心组件
# ==========================================
# 假设你的第一份文件叫 yolov8_custom.py
from yolov8 import YOLOv8, v8DetectionLoss
# 假设你的第二份文件叫 dataset.py
from dataset import YOLODataset


def yolo_collate_fn(batch):
    """
    整理函数：将多张图片和长度不一的真实标签打包成一个统一的 Batch 张量
    """
    # 提取所有的图片、标签类别、边界框
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]

    # 组合图片张量: [B, 3, 640, 640]
    imgs_tensor = torch.stack(imgs, dim=0)

    # 找到这个 Batch 中缺陷数量最多的一张图，它的缺陷数是 max_m
    max_m = max([lbl.shape[0] for lbl in labels])
    if max_m == 0:
        max_m = 1 # 兜底机制：如果整个 Batch 都是纯背景图，强制设为 1 避免后续维度崩溃

    padded_labels = []
    padded_bboxes = []

    for lbl, box in zip(labels, bboxes):
        m = lbl.shape[0]
        # 计算需要补齐的数量
        pad_size = max_m - m

        # 对类别标签进行补齐，用 -1 填充表示这是无效的占位符
        pad_lbl = torch.full((pad_size, 1), -1.0, dtype=torch.float32)
        padded_labels.append(torch.cat([lbl, pad_lbl], dim=0))

        # 对坐标框进行补齐，用 0 填充
        pad_box = torch.zeros((pad_size, 4), dtype=torch.float32)
        padded_bboxes.append(torch.cat([box, pad_box], dim=0))

    # 组合成最终的 [B, M, 1] 和 [B, M, 4]
    return imgs_tensor, torch.stack(padded_labels, dim=0), torch.stack(padded_bboxes, dim=0)


def train_model():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print("\n" + "=" * 50)
    print("🔥 引擎点火：YOLOv8 工业级训练大闭环正式启动 🔥")
    print("=" * 50)

    # 1. 实例化网络与损失结算中心
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    model = YOLOv8(nc=2).to(device)
    model.train()  # 开启训练模式

    criterion = v8DetectionLoss(nc=2, reg_max=16).to(device)

    # 2. 准备数据集与 DataLoader (直接调用导入的 YOLODataset)
    dataset = YOLODataset(img_dir="dummy_images", label_dir="dummy_labels", img_size=640)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=yolo_collate_fn)

    # 3. 配置优化器 (Optimizer)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. 进入 Epoch 循环
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, (imgs, gt_labels, gt_bboxes) in enumerate(dataloader):
            imgs = imgs.to(device)
            gt_labels = gt_labels.to(device)
            gt_bboxes = gt_bboxes.to(device)

            optimizer.zero_grad()

            # 【注意】：确保你的 YOLOv8 forward 现在返回的是两个解耦的张量
            pred_scores, pred_dist = model(imgs)

            loss = criterion(pred_scores, pred_dist, gt_labels, gt_bboxes)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"   [Epoch {epoch + 1}/{num_epochs}] Batch {batch_idx + 1} | 当前 Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch + 1} 结束 | 平均 Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == '__main__':
    train_model()