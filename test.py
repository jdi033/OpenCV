import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
from IPython.display import Image

# image = cv2.imread("checkerboard_18x18.png", cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread("checkerboard_84x84.jpg", cv2.IMREAD_GRAYSCALE)
# #cv2.imshow("image", image)
# #cv2.imshow("image", image2)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()
#
# cb_img = cv2.imread("checkerboard_18x18.png", 0)
# cb_img_fuzzy = cv2.imread("checkerboard_fuzzy_18x18.jpg", 0)
# print(cb_img)
# print(cb_img_fuzzy)
# print("Image size (H, W) is:", cb_img.shape)
# print("Data type of image is:", cb_img.dtype)
# coke_img = cv2.imread("coca-cola-logo.png", 1)
# print(coke_img)
# print("Image size (H, W, CC) is:", coke_img.shape)
# print("Data type of image is:", coke_img.dtype)
#
# #使用 Matplotlib 显示图像
# #plt.imshow(cb_img)
# #plt.imshow(cb_img, cmap="gray")
# #plt.imshow(cb_img_fuzzy, cmap="gray")
# plt.imshow(coke_img)
# coke_img_channels_reversed = coke_img[:, :, ::-1]
# plt.imshow(coke_img_channels_reversed)
# plt.show()  # 关键：显示图像窗口

img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)

# Show the channels
plt.figure(figsize=[20, 5])

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((b, g, r))
# Show the merged output
plt.subplot(144)
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")
plt.show()