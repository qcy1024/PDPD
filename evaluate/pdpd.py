import taichi as ti
import taichi.math as tm

import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import skimage
from skimage import io 
import cv2

import utils
import os

ti.init(arch=ti.gpu)

tivec3 = ti.types.vector(3, float)
mat3x3f = ti.types.matrix(3, 3, float)

img_depth_gt_path = "../raw_depth/339.png"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
img_depth_gt = img_depth_gt.astype(np.float32)
dd = 10
img_depth_gt /= dd
img_depth_gt_field = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1]))
img_depth_gt_field.from_numpy(img_depth_gt)

field_point_cloud = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1], 3))
L = 10
invalid = 0
Rough_m = 1 * 1e-2
threshhold_not_recal = 4160
INFINITY = 1.0 * 1e20
matL2x3f = ti.types.matrix(L*L, 3, float)
seedtime = 2

inv_K = np.array([[5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                    [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                    [0.000000e+00,0.000000e+00,1.000000e+00] ])
inv_K_T = np.linalg.inv(inv_K)
inv_K_T_field = ti.field(dtype=ti.f32, shape=(inv_K_T.shape[0], inv_K_T.shape[1]))
inv_K_T_field.from_numpy(inv_K_T)

def cv2_img_save(np_img, out_name, c_out_name):
    unique_planes = np.unique(np_img)
    cmap = plt.cm.get_cmap('hsv', len(unique_planes))
    color_segmented_image = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    for i, plane in enumerate(unique_planes):
        color_segmented_image[np_img == plane] = (np.array(cmap(i)[:3]) * 255).astype(int)
    cv2.imwrite(c_out_name, color_segmented_image)  
    image = Image.fromarray(np_img.astype('uint8'))
    image.save(out_name)
    return 

@ti.kernel
def depth_to_space_point():
    for i, j in img_depth_gt_field:
        p = tivec3([i+26, j+38, 1])
        D = img_depth_gt_field[i,j]
        if D != 0:
            result = tivec3([0.0,0.0,0.0])
            sum = 0.0
            for k in range(3):
                sum = 0.0
                for l in range(3):
                    sum = sum + inv_K_T_field[k,l] * p[l]
                result[k] = sum
                field_point_cloud[i,j,k] = D * result[k]
    return 

@ti.kernel
def get_seed_patch_num() -> int:
    ret = 0
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (seedtime*L) != 0 or j % (seedtime*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            ret += 1
    return ret


seed_num = get_seed_patch_num()
print("seed_num = ", seed_num)
plane_seg_img2d = ti.field(dtype=ti.i32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1]))
plane_seg_img3d = ti.field(dtype=ti.i32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1], seed_num+2))
# idx, error, a, b, d, (x,y)
plane_list = ti.Vector.field(n=8, dtype=ti.f32, shape=(seed_num+2,1))

@ti.func
def seedidx_encode(H:int, W:int, a, b):
    max_size = ti.max(H, W)
    return a * max_size + b

tivecL = ti.types.vector(L, float)
tivecL2 = ti.types.vector(L*L, float)

@ti.kernel
def get_seed_patch_seg():
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (seedtime*L) != 0 or j % (seedtime*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            idx = seedidx_encode(img_depth_gt_field.shape[0], img_depth_gt_field.shape[1], i, j)
            for k in range(L):
                for l in range(L):
                    plane_seg_img2d[i+k, j+l] = idx
    return  

@ti.kernel
def get_seed_patch_seg2():
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (seedtime*L) != 0 or j % (seedtime*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            curplane_idx = plane_seg_img2d[i, j]
            plane_list[curplane_idx, 0][0] = curplane_idx
            x_list = tivecL2([0]*(L*L))
            y_list = tivecL2([0]*(L*L))
            z_list = tivecL2([0]*(L*L))

            for k in range(L):
                for l in range(L):
                    x_list[L*k+l] = field_point_cloud[i+k, j+l, 0]
                    y_list[L*k+l] = field_point_cloud[i+k, j+l, 1]
                    z_list[L*k+l] = field_point_cloud[i+k, j+l, 2]

            X = utils.pdpd_least_square(L*L, x_list, y_list, z_list)
            a = X[0]
            b = X[1]
            d = X[2]
            err = 0.0
            sum = 0.0
            for k in range(L):
                for l in range(L):
                    normal_n = tivec3([a,b,-1])
                    p_at0 = field_point_cloud[i+k, j+l, 0]
                    p_at1 = field_point_cloud[i+k, j+l, 1]
                    p_at2 = field_point_cloud[i+k, j+l, 2]
                    ndotp_plusd = normal_n[0] * p_at0 + normal_n[1] * p_at1 + normal_n[2] * p_at2 + d
                    sum = sum + ndotp_plusd * ndotp_plusd
            sum = sum / L / L
            if sum > 0 :
                sum = tm.sqrt(sum)
            err = sum
            plane_list[curplane_idx, 0][0] = curplane_idx
            plane_list[curplane_idx, 0][1] = err 
            plane_list[curplane_idx, 0][2] = a
            plane_list[curplane_idx, 0][3] = b
            plane_list[curplane_idx, 0][4] = d
            plane_list[curplane_idx, 0][5] = i
            plane_list[curplane_idx, 0][6] = j
            plane_list[curplane_idx, 0][7] += 1
    return  

def cal_avg_seed_error():
    seed_error_avg = 0.0
    for i in range(2, seed_num+2):
        seed_error_avg += plane_list[i, 0][1]
    seed_error_avg /= seed_num
    for i in range(2, seed_num+2):
        if plane_list[i, 0][1] > seed_error_avg:
            plane_list[i, 0][1] = INFINITY
    return

def pyscope_delete_invalid_seed():
    for i in range(np_planeseg_img2d.shape[0]):
        for j in range(np_planeseg_img2d.shape[1]):
            idx = np_planeseg_img2d[i, j]
            if plane_list[idx, 0][1] >= INFINITY:
                np_planeseg_img2d[i, j] = invalid
    return 
    
@ti.func
def cal_error(n:tivec3, p:tivec3, d:float) -> float:
    return ti.abs( n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d )

@ti.func
def cal_err_T(x:int, y:int, j,tao,lambdaa,H,W,alpha,k) -> float:
    d = img_depth_gt_field[x, y]
    t = ( tao * ( 1 - tm.exp(-(j/lambdaa)) ) ) 
    ret = t * t
    if j > H * W / k / k:
        ret = ret * alpha * d * d
    return ret

@ti.kernel
def compute_seg() -> int:
    tao = 1.21
    lambdaa = 2.5
    alpha = 0.009
    kk = 61.65     
    for i in range(2, seed_num+2):
        idx = int(plane_list[i, 0][0])
        if plane_list[i, 0][1] >= INFINITY:
            continue
        x = int(plane_list[idx, 0][5])
        y = int(plane_list[idx, 0][6])
        top = 0
        down = 0
        right = 0
        left = 0
        if x - 1 >= 0:
            top = x - 1
        else:
            top = x
        if y - 1 >= 0:
            left = y - 1
        else :
            left = y
        if x + L < img_depth_gt_field.shape[0]:
            down = x + L
        else :
            down = x + L - 1
        if x + L < img_depth_gt_field.shape[1]:
            right = y + L
        else :
            right = y + L - 1
        growing = True 
        grow_stage_cnt = 1
        newtop = top
        newdown = down
        newleft = left
        newright = right

        while growing:
            last_grow_stage_cnt = grow_stage_cnt
            top = newtop
            down = newdown
            left = newleft
            right = newright
            # ****** ****** ****** code for recal ***** ****** ******
            Asum11 = 0.0
            Asum12 = 0.0
            Asum13 = 0.0
            Asum21 = 0.0
            Asum22 = 0.0
            Asum23 = 0.0
            Asum31 = 0.0
            Asum32 = 0.0
            Asum33 = 1.0
            Bsum1 = 0.0
            Bsum2 = 0.0
            Bsum3 = 0.0
            N = 0
            err_sum = 0.0
            a = plane_list[i, 0][2]
            b = plane_list[i, 0][3]
            d = plane_list[i, 0][4]
            err = plane_list[i, 0][1]
            # ***** ***** ****** code for recal end ***** ***** ***** 
            for j in range(top, down+1):
                for k in range(left, right+1):
                    a = plane_list[i, 0][2]
                    b = plane_list[i, 0][3]
                    d = plane_list[i, 0][4]
                    n = tivec3([a,b,-1])
                    if plane_seg_img3d[j, k, idx] == idx:
                        N += 1
                        xi = field_point_cloud[j, k, 0]
                        yi = field_point_cloud[j, k, 1]
                        zi = field_point_cloud[j, k, 2]
                        Asum11 += xi ** 2
                        Asum12 += xi * yi
                        Asum13 += xi
                        Asum21 += xi * yi
                        Asum22 += yi ** 2
                        Asum23 += yi
                        Asum31 += xi
                        Asum32 += yi
                        Bsum1 += xi * zi
                        Bsum2 += yi * zi
                        Bsum3 += zi
                        err_sum += ( a * xi + b * yi - zi + d ) ** 2
                        continue
                    cond1 = ( j-1 >= 0 and plane_seg_img3d[j-1, k, idx] == idx )
                    cond2 = ( j+1 <= plane_seg_img2d.shape[0] and plane_seg_img3d[j+1, k, idx] == idx )
                    cond3 = ( k-1 >= 0 and plane_seg_img3d[j, k-1, idx] == idx )
                    cond4 = ( k+1 <= plane_seg_img2d.shape[1] and plane_seg_img3d[j, k+1, idx] == idx )
                    if not cond1 and not cond2 and not cond3 and not cond4:
                        continue
                    
                    p_at0 = field_point_cloud[j, k, 0]
                    p_at1 = field_point_cloud[j, k, 1]
                    p_at2 = field_point_cloud[j, k, 2]
                    p = tivec3([p_at0, p_at1, p_at2])
                    p_err = cal_error(n,p,d)
                    err_T = cal_err_T(j, k, grow_stage_cnt,tao,lambdaa,img_depth_gt_field.shape[0],
                                      img_depth_gt_field.shape[1],alpha,kk)

                    if p_err > err_T:
                        continue
                    x = int(plane_list[i, 0][5])
                    y = int(plane_list[i, 0][6])
                    avg_x = 0.0
                    avg_y = 0.0
                    avg_z = 0.0
                    for ii in range(L):
                        for jj in range(L):
                            if x+ii >=0 and x+ii <img_depth_gt_field.shape[0] and y+jj >= 0 and y+jj < img_depth_gt_field.shape[1]:
                                avg_x += field_point_cloud[x+ii, y+jj, 0] / L / L
                                avg_y += field_point_cloud[x+ii, y+jj, 1] / L / L
                                avg_z += field_point_cloud[x+ii, y+jj, 2] / L / L

                    grow_stage_cnt += 1
                    plane_seg_img3d[j, k, idx] = idx
                    if j <= top and j > 0 : 
                        newtop = j - 1
                    if j >= down and j < img_depth_gt_field.shape[0] - 1:
                        newdown = j + 1
                    if k <= left and k > 0:
                        newleft = k - 1
                    if k >= right and k < img_depth_gt_field.shape[1] - 1:
                        newright = k + 1

                    N += 1
                    xi = field_point_cloud[j, k, 0]
                    yi = field_point_cloud[j, k, 1]
                    zi = field_point_cloud[j, k, 2]
                    Asum11 += xi ** 2
                    Asum12 += xi * yi
                    Asum13 += xi
                    Asum21 += xi * yi
                    Asum22 += yi ** 2
                    Asum23 += yi
                    Asum31 += xi
                    Asum32 += yi
                    Bsum1 += xi * zi
                    Bsum2 += yi * zi
                    Bsum3 += zi
                    err_sum += ( a * xi + b * yi - zi + d ) ** 2
                # end for k in range(left, right+1)
            # end for j in range(top, down+1)      
            if last_grow_stage_cnt == grow_stage_cnt:
                growing = False
            if grow_stage_cnt < threshhold_not_recal:
                A = mat3x3f([[Asum11, Asum12, Asum13], 
                    [Asum21, Asum22, Asum23], 
                    [Asum31, Asum32, N]])
                B = tivec3([Bsum1, Bsum2, Bsum3])
                A_ivs = A.inverse()
                X = A_ivs @ B
                err_sum /= N
                err_sum = tm.sqrt(err_sum)
                plane_list[idx, 0][1] = err_sum
                plane_list[idx, 0][2] = X[0]
                plane_list[idx, 0][3] = X[1]
                plane_list[idx, 0][4] = X[2]
        # end while growing
    return 3

evtual_seg_field = ti.field(dtype=int, shape=(plane_seg_img3d.shape[0], plane_seg_img3d.shape[1]))
avg_point_num_field = ti.Vector.field(n=8, dtype=int, shape=(plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
@ti.kernel
def merge0():
    for i, j in img_depth_gt_field:
        for k in range(2, plane_seg_img3d.shape[2]):
            idx = k
            if plane_seg_img3d[i, j, k] != k:
                continue
            avg_point_num_field[i, j][0] += plane_list[idx, 0][7]
            avg_point_num_field[i, j][1] += 1
        avg_point_num_field[i, j][0] /= avg_point_num_field[i, j][1]
    return 

@ti.kernel
def merge():
    for i, j in img_depth_gt_field:
        min_err = 2e9 + 0.0
        min_err_idx = invalid
        for k in range(2, plane_seg_img3d.shape[2]):
            if plane_seg_img3d[i, j, k] != k:
                continue
            cur_plane_idx = k
            if plane_list[cur_plane_idx, 0][7] < avg_point_num_field[i, j][0]:
                continue
            cur_err = plane_list[cur_plane_idx, 0][1]
            if min_err > cur_err:
                min_err = cur_err
                min_err_idx = k
        evtual_seg_field[i, j] = min_err_idx
    return 

@ti.kernel
def initfield():
    for i, j in plane_seg_img2d:
        plane_seg_img2d[i, j] = invalid
        avg_point_num_field[i, j][0] = 0
        avg_point_num_field[i, j][1] = 0
    return 

@ti.kernel
def initplane_seg_img3d():
    for i, j, k in plane_seg_img3d:
        plane_seg_img3d[i, j, k] = plane_seg_img2d[i, j]
    return 

def pyscope_idxmapping():
    np_planeseg_img2d = plane_seg_img2d.to_numpy()
    l = []
    for i in range(np_planeseg_img2d.shape[0]):
        for j in range(np_planeseg_img2d.shape[1]):
            if np_planeseg_img2d[i, j] not in l:
                l.append(np_planeseg_img2d[i, j])
    l = sorted(l)
    ret = {}
    idx = 1
    for val in l:
        ret[val] = idx
        idx += 1
        pass
    for i in range(np_planeseg_img2d.shape[0]):
        for j in range(np_planeseg_img2d.shape[1]):
            if np_planeseg_img2d[i, j] != invalid:
                np_planeseg_img2d[i, j] = ret[np_planeseg_img2d[i, j]]
    return np_planeseg_img2d

def print_plane_list():
    for i in range(seed_num):
        print(f"{i}: idx = {plane_list[i, 0][0]}, err = {plane_list[i, 0][1]}, a = {plane_list[i, 0][2]}\
              b = {plane_list[i, 0][3]}, d = {plane_list[i, 0][4]}, x = {plane_list[i, 0][5]}, y = {plane_list[i, 0][6]} ")
    return 

def pyscope_merge2():
    for i in range(img_depth_gt.shape[0]-1):
        for j in range(img_depth_gt.shape[1]-1):
            this_idx = np_evtual_seg[i, j]
            if this_idx == invalid:
                continue
            this_a = np_plane_list[this_idx, 0][2]
            this_b = np_plane_list[this_idx, 0][3]
            this_d = np_plane_list[this_idx, 0][4]
            right_idx = np_evtual_seg[i, j+1]
            if right_idx != invalid:
                right_a = np_plane_list[right_idx, 0][2]
                right_b = np_plane_list[right_idx, 0][3]
                right_d = np_plane_list[right_idx, 0][4]
            down_idx = np_evtual_seg[i+1, j]
            if down_idx != invalid:
                down_a = np_plane_list[down_idx, 0][2]
                down_b = np_plane_list[down_idx, 0][3]
                down_d = np_plane_list[down_idx, 0][4]
            if (right_idx != invalid and abs(this_a-right_a) < Rough_m and
                abs(this_b-right_b) < Rough_m and abs(this_d-right_d) < Rough_m) :
                np_evtual_seg[i, j+1] = this_idx
            if (down_idx != invalid and abs(this_a-down_a) < Rough_m and 
                abs(this_b-down_b) < Rough_m and abs(this_d-down_d) < Rough_m) :
                np_evtual_seg[i+1, j] = this_idx

    for i in range(img_depth_gt.shape[0]):
        for j in range(img_depth_gt.shape[1]):
            this_idx = np_evtual_seg[i, j]
            if this_idx == invalid:
                continue
            err = np_plane_list[this_idx, 0][1]
            if err > 1.4641:
                np_evtual_seg[i, j] = invalid
    return
############################################################################################################

print("inv_K_T = ",inv_K_T)

depth_to_space_point()
initfield()
get_seed_patch_seg()
np_planeseg_img2d = pyscope_idxmapping()
plane_seg_img2d.from_numpy(np_planeseg_img2d)
get_seed_patch_seg2()
np_planeseg_img2d = plane_seg_img2d.to_numpy()
cal_avg_seed_error()
pyscope_delete_invalid_seed()
plane_seg_img2d.from_numpy(np_planeseg_img2d)
seed_seg = plane_seg_img2d.to_numpy()
seed_seg_out_filename = "../our_outs_nyu/init_seeds/pdpd_seedseg_339.png"
c_seed_seg_out_filename = "../our_outs_nyu/init_seeds/pdpd_c_seedseg_339.png"

np_field_point_cloud = np.array((field_point_cloud.shape[0], field_point_cloud.shape[1], field_point_cloud.shape[2]))
np_field_point_cloud = field_point_cloud.to_numpy()

cv2_img_save(seed_seg, seed_seg_out_filename, c_seed_seg_out_filename)
print("seed seg have saved.")

initplane_seg_img3d()

start_time = time.time()
a = compute_seg()
end_time = time.time()
print("compute_seg() have finished. time cost : ",end_time - start_time )
merge()

np_plane_list = plane_list.to_numpy()
np_evtual_seg = evtual_seg_field.to_numpy()
pyscope_merge2()

plane_seg_out_filename = "../our_outs_nyu/plane_segs/pdpd_plane_seg_339.png"
c_plane_seg_out_filename = "../our_outs_nyu/plane_segs/pdpd_c_plane_seg_339.png"
cv2_img_save(np_evtual_seg, plane_seg_out_filename, c_plane_seg_out_filename)
print("evtual_seg img have saved as: ", plane_seg_out_filename)
pass
