import os
import cv2
import torch
import numpy as np
import torchvision

# 引入你的算法核心蓝图
from yolov8 import YOLOv8, make_anchor, dist2bbox


# ==========================================
# 第一部分：非极大值抑制 (NMS) 算法封装
# ==========================================
def non_max_suppression(prediction_scores, prediction_bboxes, conf_thres=0.3, iou_thres=0.45):
    """
    非极大值抑制算法 (NMS)
    Args:
        prediction_scores: 预测的类别得分张量 [N, nc] (8400, 80)
        prediction_bboxes: 预测的绝对坐标边界框 [N, 4] (8400, 4) - 格式为 x1, y1, x2, y2
        conf_thres: 置信度阈值，低于此得分的框直接抛弃
        iou_thres: 交并比阈值，重合度高于此值的同类框将被剔除
    Returns:
        保留下来的检测结果列表，每个元素格式为: [x1, y1, x2, y2, score, class_id]
    """
    # 1. 找出每个预测点在所有类别中的最大得分，以及对应的类别 ID
    # max_scores: [8400] 存储最大分数, max_class_ids: [8400] 存储类别索引
    max_scores, max_class_ids = torch.max(prediction_scores, dim=1)

    # 2. 置信度过滤：只保留最大得分超过 conf_thres 的框
    # conf_mask 是一个布尔型数组，例如 [True, False, True...]
    conf_mask = max_scores > conf_thres

    # 3. 使用掩码提取出幸存的框、得分和类别
    # 这一步瞬间就能把 8400 个框砍到可能只剩几十个
    scores = max_scores[conf_mask]
    bboxes = prediction_bboxes[conf_mask]
    class_ids = max_class_ids[conf_mask]

    # 兜底机制：如果全部框都被低分过滤掉了，直接返回空列表
    if scores.shape[0] == 0:
        return []

    # 4. 工业级 NMS 核心调用
    # 注意：torchvision.ops.batched_nms 能同时处理多类别！
    # 它的原理是给不同类别的框加上一个巨大的坐标偏移量，使得不同类别的框之间绝对不可能产生交集
    # 从而实现“只对同类别的框进行 IoU 比较和剔除”
    keep_indices = torchvision.ops.batched_nms(
        boxes=bboxes,
        scores=scores,
        idxs=class_ids,
        iou_threshold=iou_thres
    )

    # 5. 根据 NMS 返回的保留索引，提取最终的幸存者
    final_bboxes = bboxes[keep_indices]
    final_scores = scores[keep_indices]
    final_class_ids = class_ids[keep_indices]

    # 6. 将结果打包拼在一起返回：[x1, y1, x2, y2, score, class_id]
    # unsqueeze(1) 将形状从 [K] 变成 [K, 1]，方便在维度 1 上拼接
    results = torch.cat((
        final_bboxes,
        final_scores.unsqueeze(1),
        final_class_ids.unsqueeze(1).float()
    ), dim=1)

    return results.cpu().numpy()  # 转回 CPU 变成 numpy 数组，方便后续 OpenCV 画图


# ==========================================
# 第二部分：推理预处理 (模拟 Dataset 中的 Letterbox)
# ==========================================
def letterbox_image(img, new_shape=(640, 640), color=(114, 114, 114)):
    """对单张推理图片进行等比例缩放和灰边填充"""
    h_orig, w_orig = img.shape[:2]
    r = min(new_shape[0] / h_orig, new_shape[1] / w_orig)

    new_unpad_w = int(round(w_orig * r))
    new_unpad_h = int(round(h_orig * r))

    img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw = (new_shape[1] - new_unpad_w) / 2
    dh = (new_shape[0] - new_unpad_h) / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # 返回处理后的图片，以及用于后续坐标还原的缩放比例 r 和填充量 dw, dh
    return img_padded, r, dw, dh


# ==========================================
# 第三部分：主控推理程序
# ==========================================
def run_inference(image_path, weight_path, num_classes=80):
    print("启动 YOLOv8 推理引擎...")

    # 1. 挂载计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 实例化网络，并加载你历经千辛万苦炼制出来的 .pt 权重
    model = YOLOv8(nc=num_classes).to(device)
    # strict=False 允许一定的参数冗余
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    # 🚨 极其重要：将模型切入“考试模式”！
    # 关闭 BatchNorm 的均值方差更新，关闭 Dropout，使网络输出变为确定性
    model.eval()

    # 3. 读取待测试的图片
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"❌ 找不到图片: {image_path}")
        return

    # 4. 图像预处理 (Letterbox -> RGB -> 归一化 -> Tensor)
    img_padded, ratio, dw, dh = letterbox_image(orig_img, new_shape=(640, 640))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # 除以 255 归一化，维度转换 [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)

    # 模型需要接收 Batch 维度，所以用 unsqueeze 增加第 0 维: [3, 640, 640] -> [1, 3, 640, 640]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 5. 前向传播：向网络提问
    print("神经网络思考中...")
    with torch.no_grad():  # 推理不需要算梯度，省显存！
        # 在 model.eval() 模式下，你的代码返回的是 cat 在一起的 [B, nc+4, 8400]
        preds = model(img_tensor)

        # 6. 张量解码 (把网络吐出的 8400 个天文数字解开)
    # 取出第一张图的预测结果: [nc+4, 8400] -> 转置为 [8400, nc+4]
    preds = preds[0].permute(1, 0)

    # 切分出分类得分 [8400, 80] 和 边界框的相对距离 [8400, 4] (这里的4是 l,t,r,b)
    pred_scores = preds[:, :num_classes]
    pred_ltrb = preds[:, num_classes:]

    # 动态生成网格锚点 (用于将相对距离还原为绝对坐标)
    strides = [8, 16, 32]
    feats_shape = [(640 // s, 640 // s) for s in strides]

    # 🌟 接收步长张量
    anchor_points, stride_tensor = make_anchor(feats_shape, strides)
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)  # 🌟 送入 GPU

    # ==========================================
    # 🚀 绝杀修复：还原真实像素距离
    # ==========================================
    pred_ltrb = pred_ltrb * stride_tensor

    # 调用你的解码函数，将 ltrb 转换为 x1, y1, x2, y2 格式 [8400, 4]
    pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=False, dim=-1)

    # 7. 调用 NMS 非极大值抑制，砍掉多余的重叠框
    print("NMS 清理重叠框...")
    results = non_max_suppression(pred_scores, pred_bboxes, conf_thres=0.0001, iou_thres=0.45)

    if len(results) == 0:
        print("图片中没有检测到任何目标。")
        return

    # 8. 物理坐标逆向映射 & OpenCV 画图呈现
    # results 中的坐标是在 640x640 灰边图上的坐标，我们需要把它们还原回原始大图上！
    for det in results:
        x1, y1, x2, y2, score, class_id = det

        # 逆向 Letterbox 映射：减去灰边偏移量，除以缩放比例
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # 将浮点数坐标转换为整数像素点
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # 在原图上画矩形框 (使用明亮的绿色，线条粗细为 2)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 组装文本标签 (例如 "Class: 0 0.85")
        label = f"Class: {class_id} {score:.2f}"

        # 计算文本框的大小，画一个带有背景色的底框让文字更清晰
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(orig_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)  # -1 表示实心填充

        # 将白色文字写在底框上
        cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 9. 保存并展示你的丰功伟绩！
    output_path = "inference_result.jpg"
    cv2.imwrite(output_path, orig_img)
    print(f"检测完成，一共检测到 {len(results)} 个目标，结果已保存为 {output_path}")


# ==========================================
# 执行入口
# ==========================================
if __name__ == '__main__':
    # 请确保你在项目中放入了一张名为 test.jpg 的测试图片
    # 如果你跑的是 COCO 数据集，权重文件应该是 epoch_50.pt，类别数是 80
    run_inference(
        image_path="test.jpg",
        weight_path="weights/yolov8_custom_epoch_100.pt",
        num_classes=80
    )