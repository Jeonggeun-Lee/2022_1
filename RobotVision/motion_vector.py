import cv2 # cvtColor, imread, arrowedLine, imshow, waitKey
import numpy as np # gradient, stack, subtract, linalg.lstsq, linalg.norm, array, reshape, pad

neighbor_type = 0

def motion_vec(img0_src, img1_src):
    height = len(img0_src)
    width = len(img0_src[0])

    img0 = cv2.cvtColor(img0_src, cv2.COLOR_BGR2YCR_CB)
    img1 = cv2.cvtColor(img1_src, cv2.COLOR_BGR2YCR_CB)
    img0 = img0[:, :, 0]
    img1 = img1[:, :, 0]
    print('Source images')
    print('img 0')
    print(img0)
    print('img 1')
    print(img1)
    img0 = np.array(img0).astype(np.float)
    img1 = np.array(img1).astype(np.float)


    grad_y = np.gradient(img0, axis=0)
    grad_x = np.gradient(img0, axis=1)
    grad = np.stack([grad_y, grad_x], axis=-1)
    grad_t = np.subtract(img1, img0)
    print('gradient y, gradient x')
    print(grad)
    print('gradient t')
    print(grad_t)

    if neighbor_type == 0:
        grad_list = [
            grad[:-2, :-2], grad[:-2, 1:-1], grad[:-2, 2:],
            grad[1:-1, :-2], grad[1:-1, 1:-1], grad[1:-1, 2:],
            grad[2:, :-2], grad[2:, 1:-1], grad[2:, 2:]
        ]
    elif neighbor_type == 1:
        grad_list = [
            grad[:-2, 1:-1],
            grad[1:-1, :-2], grad[1:-1, 1:-1], grad[1:-1, 2:],
            grad[2:, 1:-1]
        ]

    mat_A = np.stack(grad_list, axis=2)
    print('Matrix A')
    print(mat_A)

    if neighbor_type == 0:
        grad_t_list = [
            grad_t[:-2, :-2], grad_t[:-2, 1:-1], grad_t[:-2, 2:],
            grad_t[1:-1, :-2], grad_t[1:-1, 1:-1], grad_t[1:-1, 2:],
            grad_t[2:, :-2], grad_t[2:, 1:-1], grad_t[2:, 2:]
        ]
    elif neighbor_type == 1:
        grad_t_list = [
            grad_t[:-2, 1:-1],
            grad_t[1:-1, :-2], grad_t[1:-1, 1:-1], grad_t[1:-1, 2:],
            grad_t[2:, 1:-1]
        ]

    vec_b = np.stack(grad_t_list, axis=2)
    vec_b = -vec_b
    print('Vector b')
    print(vec_b)

    m_vec_v = []
    m_vec_u = []

    for i in range(0, height-2):
        for j in range(0, width-2):
            v, u = np.linalg.lstsq(mat_A[i, j], vec_b[i, j], rcond=None)[0]
            m_vec_v.append(v)
            m_vec_u.append(u)
    m_vec_v = np.array(m_vec_v)
    m_vec_u = np.array(m_vec_u)
    m_vec_v = np.reshape(m_vec_v, (height-2, width-2))
    m_vec_u = np.reshape(m_vec_u, (height-2, width-2))
    m_vec_v = np.pad(m_vec_v, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    m_vec_u = np.pad(m_vec_u, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    m_vec = np.stack([m_vec_v, m_vec_u], axis=2)
    print('Motion Vector')
    print(m_vec)
    return m_vec


def info_change_rate_m_vec(img0_src, img1_src, m_vec):
    feat0 = np.reshape(img0_src, (-1, 3))
    feat1 = np.reshape(img1_src, (-1, 3))
    new_feat = np.concatenate([img0_src, m_vec], axis=-1)
    new_feat = np.reshape(new_feat, (-1, 5))
    mean_feat0 = np.mean(feat0, axis=0)
    mean_feat1 = np.mean(feat1, axis=0)
    mean_new_feat = np.mean(new_feat, axis=0)
    mean_squared_diff0 = np.mean(np.square(np.linalg.norm(feat0 - mean_feat0)))
    mean_squared_diff1 = np.mean(np.square(np.linalg.norm(feat1 - mean_feat1)))
    mean_mean_sd = (mean_squared_diff0+mean_squared_diff1)/2
    mean_squared_diff_new = np.mean(np.square(np.linalg.norm(new_feat - mean_new_feat)))
    return mean_squared_diff_new/mean_mean_sd


if __name__=='__main__':
    img0_path = './dataset/hop_far/frame19.jpg'
    img1_path = './dataset/hop_far/frame20.jpg'
    img0_src = cv2.imread(img0_path)
    img1_src = cv2.imread(img1_path)
    height = len(img0_src)
    width = len(img0_src[0])

    m_vec = motion_vec(img0_src, img1_src)

    threshold = 16
    color = (255, 0, 0)
    thickness = 1
    for i in range(1, height-1):
        for j in range(1, width-1):
            if 0 <= i + m_vec[i, j, 0] < height and 0 <= j + m_vec[i, j, 1] < width:
                if np.linalg.norm(m_vec[i, j]) >= threshold:
                    img_result = cv2.arrowedLine(img0_src, (i, j), (int(i + m_vec[i, j, 0]), int(j + m_vec[i, j, 1])), color, thickness)

    information_change_rate = info_change_rate_m_vec(img0_src, img1_src, m_vec)
    print('information change rate:', information_change_rate)

    cv2.imshow('motion vector', img_result)
    cv2.waitKey()
