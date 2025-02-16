import matplotlib.pyplot as plt
from PIL import Image
import skimage
from skimage import io,data
import numpy as np
import os

source_dir = "../abw_yuanshi" 
target_dir = "../abw_raw_depth" 
H, W = 512, 512

def main():
    for filename in os.listdir(source_dir):
        if filename.endswith(".range"): 
            file_path = os.path.join(source_dir, filename)  
            print(f"processing：{file_path}")
            file_size = os.path.getsize(file_path)  
            expected_size = 512 * 512 
            range_data = np.fromfile(file_path, dtype=np.int8)
            valid_range_data = range_data[:512*512].reshape(H, W)
            image = Image.fromarray(valid_range_data.astype('uint8'))
            save_file_name = filename.rsplit('.', 1)[0] + "_depth.png"
            print("save_file_name=", save_file_name)
            target_path = os.path.join(target_dir, save_file_name)
            image.save(target_path)
            print(f"saved as：{target_path}")
    return 

if __name__ == "__main__":
    main()
    return 
    