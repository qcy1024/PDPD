from PIL import Image
import skimage
from skimage import io,data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import math

def img_save(np_img,out_name,c_out_name):
    plt.figure(figsize=(np_img.shape[1]//100,np_img.shape[0]//100))
    plt.imshow(np_img, cmap='viridis', aspect='auto',interpolation="nearest")
    plt.colorbar() 
    plt.title('Initial Plane Segmentation')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(c_out_name)
    
    np_img = np.expand_dims(np_img, axis=2)
    np_img = np.repeat(np_img, repeats=3, axis=2)
    image = Image.fromarray(np_img.astype('uint8'))
    image.save(out_name)

def get_seed_patch_seg(img_depth_gt,L,point_cloud,img_obj_seg_gt):
    seed_list = {}
    seed_seg = torch.zeros((img_depth_gt.shape[0],img_depth_gt.shape[1]),dtype=torch.int32)
    i_idx = 0
    j_idx = 0
    seg_idx = 1
    while i_idx <= img_depth_gt.shape[0]-L:
        while j_idx <= img_depth_gt.shape[1]-L:
            can_be_seed_patch = True
            obj_seg = img_obj_seg_gt[i_idx][j_idx]
            for k in range(L):
                if k >= img_depth_gt.shape[0]:
                    break
                for l in range(L):
                    if l >= img_depth_gt.shape[1]:
                        break
                    if img_depth_gt[i_idx+k][j_idx+l] == 0 or seed_seg[i_idx+k][j_idx+l] != 0:
                        can_be_seed_patch = False
                        break 
                if can_be_seed_patch == False:
                    break
            if can_be_seed_patch:
                x_list = []
                y_list = []
                z_list = []
                points = []
                for k in range(L):
                    if k >= img_depth_gt.shape[0]:
                        break
                    for l in range(L):    
                        if l >= img_depth_gt.shape[1]:
                            break
                        seed_seg[i_idx+k,j_idx+l] = seg_idx
                        x_list.append(point_cloud[i_idx+k,j_idx+l,0])
                        y_list.append(point_cloud[i_idx+k,j_idx+l,1])
                        z_list.append(point_cloud[i_idx+k,j_idx+l,2])
                        points.append([i_idx+k,j_idx+l])
                seg_idx += 1
                j_idx += L  
            else :
                j_idx += 1
        j_idx = 0
        i_idx += L
    print("seed_list calculated. ")
    return seed_seg, seg_idx-1, seed_list

def pdpd_get_seed_patch_seg(img_depth_gt,L,point_cloud,img_obj_seg_gt):
    seed_list = {}
    seed_seg = torch.zeros((img_depth_gt.shape[0],img_depth_gt.shape[1]),dtype=torch.int32)
    i_idx = 0
    j_idx = 0
    seg_idx = 1
    while i_idx <= img_depth_gt.shape[0]-L:
        while j_idx <= img_depth_gt.shape[1]-L:
            
            can_be_seed_patch = True
            obj_seg = img_obj_seg_gt[i_idx][j_idx]
            for k in range(L):
                if k >= img_depth_gt.shape[0]:
                    break
                for l in range(L):
                    if l >= img_depth_gt.shape[1]:
                        break
                    if img_depth_gt[i_idx+k][j_idx+l] == 0 or seed_seg[i_idx+k][j_idx+l] != 0 or img_obj_seg_gt[i_idx, j_idx] != img_obj_seg_gt[i_idx+k, j_idx+l]:
                        can_be_seed_patch = False
                        break 
                    if i_idx % (2*L) != 0 or j_idx % (2*L) != 0:
                        can_be_seed_patch = False
                        break
                if can_be_seed_patch == False:
                    break
            if can_be_seed_patch:
                x_list = []
                y_list = []
                z_list = []
                points = []
                for k in range(L):
                    if k >= img_depth_gt.shape[0]:
                        break
                    for l in range(L):    
                        if l >= img_depth_gt.shape[1]:
                            break
                        seed_seg[i_idx+k,j_idx+l] = seg_idx
                        x_list.append(point_cloud[i_idx+k,j_idx+l,0])
                        y_list.append(point_cloud[i_idx+k,j_idx+l,1])
                        z_list.append(point_cloud[i_idx+k,j_idx+l,2])
                        points.append([i_idx+k,j_idx+l])
                seg_idx += 1
                j_idx += L  
            else :
                j_idx += 1
        j_idx = 0
        i_idx += L
    print("seed_list calculated. ")
    return seed_seg, seg_idx-1, seed_list

def depth_to_space_point(img_depth_gt,inv_K_T):
    if isinstance(img_depth_gt, np.ndarray):
        img_depth_gt = torch.from_numpy(img_depth_gt)
    if isinstance(inv_K_T, np.ndarray):
        inv_K_T = torch.from_numpy(inv_K_T)
    point_cloud = torch.zeros((img_depth_gt.shape[0],img_depth_gt.shape[1],3))
    for i in range(img_depth_gt.shape[0]):
        for j in range(img_depth_gt.shape[1]):
            p = torch.tensor([i,j,1],dtype=torch.float32)
            D = img_depth_gt[i,j]
            if D != 0:
                point_cloud[i,j] = D * torch.mv(inv_K_T,p)
    return point_cloud

if __name__ == "__main__":
    img_rgb_path = "26_rgb.jpg"
    img_rgb = skimage.io.imread(img_rgb_path)
    img_rgb = img_rgb.astype(np.float32)
    img_depth_gt_path = "26_depth_gt.png"
    img_depth_gt = skimage.io.imread(img_depth_gt_path)
    img_depth_gt = img_depth_gt.astype(np.float32)
    dd = 1
    img_depth_gt /= dd
    img_objseg_gt_name = "26_obj_seg.png"
    img_objseg_gt = skimage.io.imread(img_objseg_gt_name)
    L = 10

    inv_K = torch.tensor( [ [5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                            [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                            [0.000000e+00,0.000000e+00,1.000000e+00] ])
    inv_K_T = torch.inverse(inv_K)

    point_cloud = depth_to_space_point(img_depth_gt, inv_K_T)

    split_image, seed_num, _ = get_seed_patch_seg(img_depth_gt, L, point_cloud, img_objseg_gt) 
    split_image = split_image.numpy()
    unique_planes = np.unique(split_image)

    cmap = plt.cm.get_cmap('hsv', len(unique_planes)) 

    color_segmented_image = np.zeros((split_image.shape[0], split_image.shape[1], 3), dtype=np.uint8)
    for i, plane in enumerate(unique_planes):
        color_segmented_image[split_image == plane] = (np.array(cmap(i)[:3]) * 255).astype(int)

    cv2.imwrite('342_dpd_seed.png', color_segmented_image) 
    pdpd_split_image, pdpd_seed_num, _ = pdpd_get_seed_patch_seg(img_depth_gt, L, point_cloud, img_objseg_gt)  
    pdpd_split_image = pdpd_split_image.numpy()
    pdpd_unique_planes = np.unique(pdpd_split_image)
    pdpd_cmap = plt.cm.get_cmap('hsv', len(pdpd_unique_planes)) 
    pdpd_color_segmented_image = np.zeros((pdpd_split_image.shape[0], pdpd_split_image.shape[1], 3), dtype=np.uint8)
    for i, plane in enumerate(pdpd_unique_planes):
        pdpd_color_segmented_image[pdpd_split_image == plane] = (np.array(pdpd_cmap(i)[:3]) * 255).astype(int)

    cv2.imwrite('342_pdpd_seed.png', pdpd_color_segmented_image) 
    pass

