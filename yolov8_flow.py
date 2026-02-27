import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(20, 14))

# ==================== 左图：模块层级调用关系 ====================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
ax1.set_title('YOLOv8 模块层级调用关系', fontsize=14, weight='bold', pad=20)

# 颜色定义
colors = {
    'model': '#E3F2FD',
    'backbone': '#BBDEFB',
    'neck': '#C8E6C9',
    'head': '#FFCCBC',
    'loss': '#FFCDD2',
    'component': '#FFF9C4',
    'base': '#F5F5F5'
}


def draw_module(ax, x, y, w, h, color, name, desc, fontsize=9):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.3, name, ha='center', va='top',
            fontsize=fontsize + 1, weight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.1, desc, ha='center', va='center',
            fontsize=fontsize - 1, color='#333333')


def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=2, style='->'):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# 顶层：YOLOv8模型
draw_module(ax1, 3.5, 12.5, 3, 1, colors['model'],
            'YOLOv8 (主模型)', '整合Backbone+Neck+Head\n管理整体流程', 10)

# 第二层：三大组件
draw_module(ax1, 0.5, 10, 2.5, 1.2, colors['backbone'],
            'Backbone', '特征提取\n输出P3/P4/P5', 9)
draw_module(ax1, 3.75, 10, 2.5, 1.2, colors['neck'],
            'Neck (PANet)', '特征融合\n双向路径增强', 9)
draw_module(ax1, 7, 10, 2.5, 1.2, colors['head'],
            'YOLOv8Head', '检测预测\n3尺度解耦头', 9)

# 连接顶层到第二层
draw_arrow(ax1, 5, 12.5, 1.75, 11.2)
draw_arrow(ax1, 5, 12.5, 5, 11.2)
draw_arrow(ax1, 5, 12.5, 8.25, 11.2)

# Backbone子模块
backbone_modules = [
    ('Conv (Stem)', '下采样\n640→160', 0.2, 8),
    ('C2f×3', 'Stage 1\n160×160', 0.2, 6.5),
    ('C2f×6', 'Stage 2 (P3)\n80×80', 0.2, 5),
    ('C2f×6', 'Stage 3 (P4)\n40×40', 0.2, 3.5),
    ('C2f×3', 'Stage 4\n20×20', 0.2, 2),
    ('SPPF', '空间金字塔\n20×20', 0.2, 0.5),
]
for name, desc, x, y in backbone_modules:
    draw_module(ax1, x, y, 2, 1.2, colors['component'], name, desc, 8)
    if y > 0.5:
        draw_arrow(ax1, x + 1, y + 1.2, x + 1, y + 0.2, lw=1.5)

# 连接Backbone到子模块
draw_arrow(ax1, 1.75, 10, 1.2, 9.2, lw=1.5)

# Neck子模块
neck_modules = [
    ('Upsample×2', '上采样\n20→40', 3.5, 8),
    ('Concat+C2f', '融合P4\n40×40', 3.5, 6.5),
    ('Upsample×2', '上采样\n40→80', 3.5, 5),
    ('Concat+C2f', '融合P3\n80×80 (N3)', 3.5, 3.5),
    ('Conv+C2f', '下采样\n80→40 (N4)', 3.5, 2),
    ('Conv+C2f', '下采样\n40→20 (N5)', 3.5, 0.5),
]
for i, (name, desc, x, y) in enumerate(neck_modules):
    draw_module(ax1, x, y, 2, 1.2, colors['component'], name, desc, 8)
    if i > 0:
        if i < 4:  # 上采样路径
            draw_arrow(ax1, x + 1, y + 1.2, x + 1, y + 0.2, lw=1.5)
        else:  # 下采样路径
            prev_y = [8, 6.5, 5, 3.5, 2][i - 1] if i < 5 else 2
            draw_arrow(ax1, x + 1, prev_y, x + 1, y + 1.2, lw=1.5)

# 连接Neck到子模块
draw_arrow(ax1, 5, 10, 4.5, 9.2, lw=1.5)

# Head子模块
head_modules = [
    ('DecoupledHead', 'P3检测头\n80×80', 6.8, 8),
    ('DecoupledHead', 'P4检测头\n40×40', 6.8, 6),
    ('DecoupledHead', 'P5检测头\n20×20', 6.8, 4),
]
for name, desc, x, y in head_modules:
    draw_module(ax1, x, y, 2.2, 1.5, colors['component'], name,
                desc + '\nCls+Reg分支', 8)

# 解耦头内部结构
for i, y_base in enumerate([8, 6, 4]):
    # Cls分支
    draw_module(ax1, 9.2, y_base + 0.6, 0.8, 0.8, colors['base'],
                'Cls', 'Conv×2\n1×1', 7)
    # Reg分支
    draw_module(ax1, 9.2, y_base - 0.2, 0.8, 0.8, colors['base'],
                'Reg', 'Conv×2\nDFL', 7)
    # 连接线
    draw_arrow(ax1, 9, y_base + 1, 9.2, y_base + 1.2, lw=1)
    draw_arrow(ax1, 9, y_base + 1, 9.2, y_base + 0.6, lw=1)

# 连接Head到子模块
draw_arrow(ax1, 8.25, 10, 7.9, 9.5, lw=1.5)

# 添加图例说明
ax1.text(0.2, 13.5, '【层级说明】', fontsize=10, weight='bold')
ax1.text(0.2, 13.1, '• 顶层: YOLOv8主类，管理整体流程', fontsize=8)
ax1.text(0.2, 12.8, '• 中层: Backbone/Neck/Head三大组件', fontsize=8)
ax1.text(0.2, 12.5, '• 底层: 具体模块实现（C2f、Conv等）', fontsize=8)

# ==================== 右图：数据流与调用时序 ====================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis('off')
ax2.set_title('数据流与模块调用时序', fontsize=14, weight='bold', pad=20)

# 时间轴
ax2.plot([1, 1], [0.5, 13.5], 'k-', lw=2)
ax2.text(0.5, 13.7, '调用时序', fontsize=10, weight='bold', rotation=90, va='bottom')

# 绘制时序步骤
steps = [
    (13, 'Input', '输入图像\n(batch, 3, 640, 640)', colors['model']),
    (12, 'YOLOv8.forward()', '调用主模型前向', colors['model']),
    (11, 'Backbone.forward()', '特征提取阶段', colors['backbone']),
    (10.2, '  ├─ Conv×2 (Stem)', '640→160 下采样', colors['component']),
    (9.4, '  ├─ C2f×3', 'Stage 1 特征提取', colors['component']),
    (8.6, '  ├─ C2f×6 (P3)', '输出80×80特征', colors['component']),
    (7.8, '  ├─ C2f×6 (P4)', '输出40×40特征', colors['component']),
    (7, '  └─ SPPF (P5)', '输出20×20特征', colors['component']),
    (6, 'Neck.forward(P3,P4,P5)', '特征融合阶段', colors['neck']),
    (5.2, '  ├─ Upsample + Concat', '自顶向下路径', colors['component']),
    (4.4, '  └─ Conv + Concat', '自底向上路径', colors['component']),
    (3.5, 'Head.forward(N3,N4,N5)', '检测预测阶段', colors['head']),
    (2.7, '  ├─ DecoupledHead (P3)', '80×80 解耦预测', colors['component']),
    (1.9, '  ├─ DecoupledHead (P4)', '40×40 解耦预测', colors['component']),
    (1.1, '  └─ DecoupledHead (P5)', '20×20 解耦预测', colors['component']),
    (0.2, 'Output', '返回3个尺度预测', colors['model']),
]

for y, name, desc, color in steps:
    # 绘制节点
    circle = plt.Circle((1, y), 0.15, color=color, ec='black', linewidth=2, zorder=5)
    ax2.add_patch(circle)

    # 绘制模块框
    box = FancyBboxPatch((1.5, y - 0.3), 3, 0.6, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax2.add_patch(box)
    ax2.text(3, y + 0.05, name, ha='center', va='center', fontsize=9, weight='bold')

    # 描述
    ax2.text(5, y, desc, ha='left', va='center', fontsize=8, color='#333')

    # 连接线（除了第一个）
    if y < 13:
        ax2.plot([1, 1], [y + 0.15, y + 0.7], 'k-', lw=2)

# 添加关键特征说明
features = [
    (7.5, '关键特征1: C2f模块', '• Split操作分流梯度\n• 多Bottleneck特征重用\n• 比C3更轻量高效'),
    (5, '关键特征2: PANet结构', '• 双向特征融合\n• 自顶向下传递语义\n• 自底向上传递位置'),
    (2.5, '关键特征3: 解耦头', '• 分类/回归分离\n• Anchor-Free设计\n• DFL分布回归'),
]

for y, title, content in features:
    # 高亮框
    highlight = FancyBboxPatch((6.5, y - 1), 3.3, 1.8, boxstyle="round,pad=0.02",
                               facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2, linestyle='--')
    ax2.add_patch(highlight)
    ax2.text(8.15, y + 0.5, title, ha='center', va='top', fontsize=9, weight='bold', color='#BF360C')
    ax2.text(6.7, y + 0.1, content, ha='left', va='top', fontsize=7.5, color='#333')

plt.tight_layout()
plt.savefig('/yolov8_code_flow.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("YOLOv8模块调用流程图已保存")