from PIL import Image
import skimage
from skimage import io,data
import numpy as np

img_depth_gt_path = "Scene1.bmp"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
img_depth_gt = img_depth_gt.astype(np.float32)

unique_values = np.unique(img_depth_gt)
sorted_values = np.sort(unique_values)

value_map = {sorted_values[i]: sorted_values[-(i+1)] for i in range(len(sorted_values))}
mapped_array = np.vectorize(value_map.get)(img_depth_gt)

print("source：\n", img_depth_gt)
print("mapped：\n", mapped_array)
