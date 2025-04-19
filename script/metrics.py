import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise_distances
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind

def is_the_unique_min_num(num, num_list):
    return num == min(num_list) and list(num_list).count(num) == 1

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cos_distance(a, b):
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - cosine_similarity



def nearest_neighbor_analysis(num_z_c, num_labels, metric='euclidean'):
    """
    paramters:
    ----------------
    num_z_c: list of embeddings
    num_labels: list of number labels
    metric: str, optional, default 'euclidean'

    return:
    ----------------
    valid_nearest_neighbor_rate: the score of orderness
    """
    ascend_sorted_num_z, ascend_sorted_label = ascend_sort_embs(num_z_c, num_labels)
    distance_matrix = pairwise_distances(ascend_sorted_num_z, Y=None, metric=metric)
    valid_nearest_neighbor = 0
    total_num = 0
    for i in range(len(ascend_sorted_label)):
      left_neighbor_idx = i-1
      if left_neighbor_idx >= 0:
        total_num += 1
        left_neighbor_distance = distance_matrix[i, left_neighbor_idx]
        all_left_distances = distance_matrix[i, :i]
        if is_the_unique_min_num(left_neighbor_distance, all_left_distances):
          valid_nearest_neighbor += 1
      right_neighbor_idx = i+1
      if right_neighbor_idx <= len(ascend_sorted_label)-1:
        total_num += 1
        right_neighbor_distance = distance_matrix[i, right_neighbor_idx]
        all_right_distances = distance_matrix[i, i+1:]
        if is_the_unique_min_num(right_neighbor_distance, all_right_distances):
          valid_nearest_neighbor += 1
    return valid_nearest_neighbor / total_num

import matplotlib.pyplot as plt
from matplotlib import collections as matcoll


COLOR_LIST = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'hotpink', 'gray', 'steelblue', 'olive']


def plot_num_position_in_two_dim_repr(num_z, num_labels, fig_name, x_limit=None, y_limit=None, all_embs=None):
    plt.figure(figsize=(5, 5))
    assert len(num_z[0]) == 2, f"The representation dimension of a number should be two, but got {len(num_z[0])} instead."
    sorted_label = sorted(num_labels)
    sorted_indices = [i[0] for i in sorted(enumerate(num_labels), key=lambda x: x[1])]
    sorted_num_z = [num_z[i] for i in sorted_indices]
    X = [item[0] for item in sorted_num_z]
    Y = [item[1] for item in sorted_num_z]
    max_repeating_num = find_most_frequent_elements_repeating_num(num_labels)
    for i in range(0, len(num_z)):
        plt.scatter(X[i], Y[i],
                    marker=f'${sorted_label[i]}$',
                    s=200,
                    alpha=min(1, 1/max_repeating_num*1.3),
                    c=COLOR_LIST[sorted_label[i] % len(COLOR_LIST)])
        if all_embs is None:
            plt.grid(True)
    plt.plot(X, Y, linestyle='dashed', linewidth=0.5)
    if all_embs is not None:
        embs_x = [item[0] for item in all_embs]
        embs_y = [item[1] for item in all_embs]
        plt.scatter(embs_x, embs_y, marker='o', s=1, c='navy')
    plt.xlabel('z1')
    plt.ylabel('z2')
    if x_limit is not None:
        plt.xlim(x_limit[0], x_limit[1])
    if y_limit is not None:
        plt.ylim(y_limit[0], y_limit[1])
    if x_limit is None and y_limit is None:
        plt.axis('equal')

    plt.show()
    plt.savefig(fname=fig_name)


def find_most_frequent_elements_repeating_num(arr):
    nd_array = np.array(arr)
    unique_elements, counts = np.unique(nd_array, return_counts=True)
    max_count = np.max(counts)
    return max_count

def distance_linear_analysis(embs, labels):
    """
    paramters:
    ----------------
    embs: list of embeddings
    labels: list of number labels

    return:
    ----------------
    r2: the score of orderness
    k: the slope of the linear regression
    b: the intercept of the linear regression
    """
    dis_list, sort_label = gen_dis_list(embs, labels)
    r2, k, b = distance_linear_regression(dis_list, sort_label)
    return r2, k, b
    
def ascend_sort_embs(embs, labels):
    assert len(embs) == len(labels), f"embs and labels should have the same length, but got {len(embs)} and {len(labels)} instead."
    ascend_sorted_label = sorted(np.array(labels))
    ascend_sorted_indices = np.array([i[0] for i in sorted(enumerate(labels), key=lambda x: x[1])])
    ascend_sorted_num_z = np.array([embs[i] for i in ascend_sorted_indices])
    return ascend_sorted_num_z, ascend_sorted_label


def gen_dis_list(embs, labels):
    sort_emb, sort_label = ascend_sort_embs(embs, labels)
    dis_list = []
    for i in range(0, len(embs)-1):
        dis = np.linalg.norm(sort_emb[i] - sort_emb[i+1])
        dis_list.append(dis)
    return dis_list, sort_label[0:-1]


def distance_linear_regression(dists, labels):
    X = np.array(labels).reshape(-1, 1)
    y = np.array(dists)
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    k = model.coef_[0]
    b = model.intercept_
    return r2, k, b
