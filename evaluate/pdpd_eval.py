import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import skimage
from skimage import io,data

from collections import Counter

def evaluate(plane_labeled_gt, out, gt_F, out_F):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    sensitivity = 0.0
    specificity = 0.0
    plane_idx_list = get_dpd_nyu_plane_list_gt(plane_labeled_gt)
    for i in range(plane_labeled_gt.shape[0]):
        for j in range(plane_labeled_gt.shape[1]):
            if plane_labeled_gt[i, j] != gt_F and out[i, j] != out_F :
                TP += 1
            elif plane_labeled_gt[i, j] != gt_F and out[i, j] == out_F:
                FN += 1
            elif plane_labeled_gt[i, j] == gt_F and out[i, j] != out_F:
                FP += 1
            elif plane_labeled_gt[i, j] == gt_F and out[i, j] == out_F:
                TN += 1
        pass
    pass
    sensitivity = TP / ( TP + FN )
    specificity = TN / ( TN + FP )
    print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}")
    return sensitivity, specificity


def cal_CDR(Sgt, S, threshold=0.8):
    assert Sgt.shape == S.shape,
    plane_ids_gt = np.unique(Sgt)
    total_planes = len(plane_ids_gt)
    detected_planes = 0

    for plane_id in plane_ids_gt:
        mask_gt = (Sgt == plane_id)
        true_plane_pixels = np.sum(mask_gt) 
        if true_plane_pixels == 0:
            continue
        
        predicted_ids = S[mask_gt]
        predicted_count = Counter(predicted_ids)
        best_match_count = 0
        for predicted_id, count in predicted_count.items():
            if predicted_id != 0:  
                best_match_count = max(best_match_count, count)
        if best_match_count / true_plane_pixels >= threshold:
            detected_planes += 1
    
    CDR = detected_planes / total_planes if total_planes > 0 else 0
    return CDR

if __name__ == "__main__":
    Sgt = skimage.io.imread("../plane_labeled_gt/depth_761_GT.bmp")
    Srgb = skimage.io.imread("../our_outs_nyu/plane_segs/pdpd_plane_seg_761.png")
    S = np.zeros((Srgb.shape[0], Srgb.shape[1]))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            S[i, j] = Srgb[i, j]

    sensitivity, specificity = evaluate(Sgt, S, 0, 0)

    CDR = cal_CDR(Sgt, S)
    print(f"sensitivity:{sensitivity}, specificity:{specificity}, CDR: {CDR}")
    pass
    