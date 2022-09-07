import cv2 # imread, imshow, waitKey
import numpy as np # linalg.norm, zeros_like, exp, stack, sum, array, reshape
from motion_vector import motion_vec # self defined
import itertools # product

def sd_change_rate_mean_shift(feat, cent):
    height = len(feat)
    width = len(feat[0])
    norm_mat_list = []
    for i in cent.keys():
        feature_diff = feat - feat[int(cent[i][3])][int(cent[i][4])]
        norm_mat_list.append(np.linalg.norm(feature_diff, axis=-1))
    norm_mat = np.stack(norm_mat_list, axis=-1)

    diff_from_cent = np.min(norm_mat, axis=-1)

    squared_diff_from_cent = diff_from_cent * diff_from_cent
    mean_sd_from_cent = np.sum(squared_diff_from_cent)/(height*width)

    feat = np.reshape(feat, (-1, feat.shape[-1]))
    mean_feat = np.mean(feat, axis=0)
    feature_diff = feat - mean_feat
    norm_arr = np.linalg.norm(feature_diff, axis=-1)
    squared_diff = norm_arr*norm_arr
    mean_sd = np.mean(squared_diff)

    print('mean of squared difference:', mean_sd)
    print('mean of squared difference from cluster center:', mean_sd_from_cent)
    print('squared difference change rate:', mean_sd_from_cent/mean_sd)
    return mean_sd_from_cent/mean_sd

def gaussian_kernel(x):
    norm = np.linalg.norm(x, axis=-1)
    gk = np.zeros_like(norm)
    gk[norm <= 1] = np.exp(-norm[norm <= 1]*norm[norm <= 1])
    return gk

def y_next(feature_vec, y):
    kernel = gaussian_kernel((feature_vec[:, 0:3]-y[0:3])/hr)\
            *gaussian_kernel((feature_vec[:, 3:5]-y[3:5])/hs)\
            *gaussian_kernel((feature_vec[:, 5:7]-y[5:7])/hm)
    kernel_augmented = np.stack([kernel]*7, axis=-1)
    return np.sum(feature_vec*kernel_augmented, axis=0) / np.sum(kernel)

def dist_tuple(z, v):
    bgr_dist = np.linalg.norm(z[0:3]-v[0:3])
    spatial_dist = np.linalg.norm(z[3:5]-v[3:5])
    motion_dist = np.linalg.norm(z[5:7] - v[5:7])
    return bgr_dist, spatial_dist, motion_dist

hr = 32
hs = 32
hm = 64
convergence_threshold = 32
min_area = 16
img0_path = './dataset/hop_far/frame19.jpg'
img1_path = './dataset/hop_far/frame20.jpg'

img0_src = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1_src = cv2.imread(img1_path, cv2.IMREAD_COLOR)
height = len(img0_src)
width = len(img0_src[0])
coord = np.array(list(itertools.product(range(height), range(width))))
coord = np.reshape(coord, (height, width, 2))
m_vec = motion_vec(img0_src, img1_src)

feature_mat = np.concatenate([img0_src, coord, m_vec], axis=-1)
feature_vec = np.reshape(feature_mat, (height*width, 7))

print('mode seeking')
v = []
for i in range(height*width):
    y = feature_vec[i]
    while True:
        y_n = y_next(feature_vec, y)
        if np.linalg.norm(y_n-y) <= convergence_threshold:
            print('pixel '+str(i)+' convergence point:')
            print(y_n)
            break
        y = y_n
    v.append(y_n)

print('clustering')
num_clusters = 0
clusters = {}
cluster_centers = {}
belongs_to = np.zeros((height*width,), dtype=np.int)

for i in range(len(v)):
    new_cluster_flag = True
    for j in range(num_clusters):
        bgr_d, spatial_d, motion_d = dist_tuple(v[i], cluster_centers[j])
        if bgr_d <= hr and spatial_d <= hs and motion_d <= hm:
            new_cluster_flag = False
            clusters[j].append(v[i])
            belongs_to[i] = j
            break
    if new_cluster_flag:
        clusters[num_clusters] = []
        clusters[num_clusters].append(v[i])
        cluster_centers[num_clusters] = v[i]
        belongs_to[i] = num_clusters
        num_clusters += 1
    print('convergence point ', str(i), ' belongs to cluster ', belongs_to[i])

print('small segment removal')
for i in range(num_clusters):
    if i not in clusters.keys():
        continue
    if len(clusters[i]) < min_area:
        print('Cluster', str(i),'is small=>removing')
        min_dist = 10000000000000
        nearest_cluster = -1
        for j in range(num_clusters):
            if j not in clusters.keys():
                continue
            if j != i:
                center_diff = np.linalg.norm(cluster_centers[j]-cluster_centers[i])
                if center_diff < min_dist:
                    min_dist = center_diff
                    nearest_cluster = j

        for feat in clusters[i]:
            belongs_to[int(feat[3])*height+int(feat[4])] = nearest_cluster
        clusters[nearest_cluster] = clusters[nearest_cluster] + clusters[i]
        del(clusters[i])
        del(cluster_centers[i])

for i in clusters.keys():
    print('cluster ', str(i), 'size:')
    print(len(clusters[i]))
    print('center ', str(i), ':')
    print(cluster_centers[i])
for i in range(len(belongs_to)):
    print('pixel ', str(i), ' belongs to cluster ', belongs_to[i])

belongs_to = np.reshape(belongs_to, (height, width))
img = np.zeros((3, belongs_to.shape[0], belongs_to.shape[1]))
img[0][belongs_to % 3 == 0] = (num_clusters - belongs_to[belongs_to % 3 == 0])*255//num_clusters
img[2][belongs_to % 3 == 1] = (num_clusters - belongs_to[belongs_to % 3 == 1])*255//num_clusters
img[1][belongs_to % 3 == 2] = (num_clusters - belongs_to[belongs_to % 3 == 2])*255//num_clusters
img = np.transpose(img, (1, 2, 0))
img = img.astype(np.uint8)

sd_change_rate = sd_change_rate_mean_shift(feature_mat, cluster_centers)

cv2.imshow('segmented image', img)
cv2.waitKey()
