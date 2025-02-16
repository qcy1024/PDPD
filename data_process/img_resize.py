import numpy as np
from PIL import Image
import skimage
from skimage import io,data
import os

def raw_rgb_resize(raw_rgb):
    ret = np.zeros((427, 564))
    for i in range(26, 453):
        for j in range(38, 602):
            for k in range(3):
                ret[i-26, j-38] = raw_rgb[i, j]
    return ret

source_dir = "../d_raw_obj_seg" 
target_dir = "../raw_obj_seg"

def main():
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"): 
            file_path = os.path.join(source_dir, filename) 
            print(f"processing：{file_path}")
            img = skimage.io.imread(file_path)
            img = img.astype(np.float32)

            ret = raw_rgb_resize(img)
            image = Image.fromarray(ret.astype('uint8'))
            target_path = os.path.join(target_dir, filename)
            image.save(target_path)
            print(f"saved as：{target_path}")
    return 

if __name__ == "__main__":
    main()
    return 