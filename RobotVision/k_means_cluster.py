import cv2 # imread, imshow, waitKey
import numpy as np # concatenate, stack, argmin, min, sum, zeros, transpose
from motion_vector import motion_vec #self defined
import itertools # product
import math # sqrt

def sd_change_rate_k_means(feat, cent):
    num_cents = len(cent)
    norm_mat_list = []
    for i in range(num_cents):
        feature_diff = feat - cent[i]
        norm_mat_list.append(np.linalg.norm(feature_diff, axis=-1))
    norm_mat = np.stack(norm_mat_list, axis=-1)
    diff_from_cent = np.min(norm_mat, axis=-1)

    squared_diff_from_cent = diff_from_cent * diff_from_cent
    mean_sd_from_cent = np.mean(squared_diff_from_cent)

    feat = np.reshape(feat, (-1, feat.shape[-1]))
    mean_feat = np.mean(feat, axis=0)
    feature_diff = feat - mean_feat
    norm_arr = np.linalg.norm(feature_diff, axis=-1)
    squared_diff = norm_arr*norm_arr
    mean_sd = np.mean(squared_diff)

    print('mean of squared difference:', mean_sd)
    print('mean of squared difference from centroid:', mean_sd_from_cent)
    print('squared difference change rate:', mean_sd_from_cent/mean_sd)
    return mean_sd_from_cent/mean_sd

num_clusters = 36
convergence_threshold = 16

img0_path = './dataset/hop_far/frame19.jpg'
img1_path = './dataset/hop_far/frame20.jpg'
img0_src = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1_src = cv2.imread(img1_path, cv2.IMREAD_COLOR)
height = len(img0_src)
width = len(img0_src[0])
m_vec = motion_vec(img0_src, img1_src)
img_feature = np.concatenate([img0_src, m_vec], axis=2)
print('Object Features')
print(img_feature)

square_root = int(math.sqrt(num_clusters))
cent_y = [(i+1)*height//square_root for i in range(square_root-1)]
cent_x = [(i+1)*width//square_root for i in range(square_root-1)]
cent = np.stack([img_feature[i, j] for j in cent_x for i in cent_y], axis=0)
num_cents = len(cent)
print('Initial Centroids')
print(cent)

iter = -1
epsilon = 0
prev_mean_sd_from_center = 0
while True:
    iter += 1
    print('iter:', iter)
    norm_mat_list = []
    for i in range(num_cents):
        feature_diff = img_feature - cent[i]
        norm_mat_list.append(np.linalg.norm(feature_diff, axis=-1))

    norm_mat = np.stack(norm_mat_list, axis=-1)
    belongs_to = np.argmin(norm_mat, axis=-1)
    diff_from_cent = np.min(norm_mat, axis=-1)

    squared_diff_from_cent = diff_from_cent * diff_from_cent
    mean_sd_from_cent = np.sum(squared_diff_from_cent) / (height * width)

    if iter == 0:
        epsilon = mean_sd_from_cent
    else:
        epsilon = abs(prev_mean_sd_from_center - mean_sd_from_cent)
    if epsilon <= convergence_threshold:
        break
    prev_mean_sd_from_center = mean_sd_from_cent
    new_cent = []
    for i in range(num_cents):
        y_cords, x_cords = np.where(belongs_to == i)
        belonged_features = np.stack([img_feature[y_cords[i], x_cords[i]] for i in range(len(y_cords))], axis=0)
        temp_cent = np.mean(belonged_features, axis=0)
        new_cent.append(temp_cent)
    cent = np.stack(new_cent, axis=0)
    print('Centroids during Iteration: ', str(iter))
    print(cent)

img = np.zeros((3, belongs_to.shape[0], belongs_to.shape[1]))
img[0][belongs_to % 3 == 0] = (num_cents - belongs_to[belongs_to % 3 == 0])*255//num_cents
img[2][belongs_to % 3 == 1] = (num_cents - belongs_to[belongs_to % 3 == 1])*255//num_cents
img[1][belongs_to % 3 == 2] = (num_cents - belongs_to[belongs_to % 3 == 2])*255//num_cents
img = np.transpose(img, (1, 2, 0))
img = img.astype(np.uint8)

sd_change_rate = sd_change_rate_k_means(img_feature, cent)

cv2.imshow('segmented image', img)
cv2.waitKey()
