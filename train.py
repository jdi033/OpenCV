# train.py (新建的总控文件)
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os

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

    # 【改动 1】：COCO 数据集有 80 个类别，必须把 nc 改为 80！
    NUM_CLASSES = 80
    model = YOLOv8(nc=NUM_CLASSES).to(device)
    model.train()

    criterion = v8DetectionLoss(nc=NUM_CLASSES, reg_max=16).to(device)

    # 【改动 2】：对接真实的 COCO128 文件夹路径，并将 Batch Size 调大一点利用 GPU (例如 4 或 8)
    dataset = YOLODataset(
        img_dir="coco128/images/train2017",
        label_dir="coco128/labels/train2017",
        img_size=640
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=yolo_collate_fn)

    # 3. 配置优化器 (Optimizer)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 【改动 3】：创建存放权重的文件夹，并将 Epoch 数量调大
    os.makedirs("weights", exist_ok=True)
    num_epochs = 50

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, (imgs, gt_labels, gt_bboxes) in enumerate(dataloader):
            imgs = imgs.to(device)
            gt_labels = gt_labels.to(device)
            gt_bboxes = gt_bboxes.to(device)

            optimizer.zero_grad()

            pred_scores, pred_dist = model(imgs)

            loss = criterion(pred_scores, pred_dist, gt_labels, gt_bboxes)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 打印每个 batch 的 loss 看得太眼花，可以注释掉，只看 Epoch 的平均 Loss
            # print(f"   [Epoch {epoch + 1}/{num_epochs}] Batch {batch_idx + 1} | 当前 Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1}/{num_epochs} 结束 | 平均 Loss: {avg_loss:.4f}")

        # 【改动 4】：每 10 个 Epoch，以及最后 1 个 Epoch 时，保存模型的“灵魂”！
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            save_path = f"weights/yolov8_custom_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型权重已保存至: {save_path}")

    print("\n🎉 训练圆满结束！你的模型已经学到了知识！")


if __name__ == '__main__':
    train_model()