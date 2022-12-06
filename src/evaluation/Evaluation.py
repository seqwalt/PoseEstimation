import numpy as np
from scipy.spatial.transform import Rotation as R

def translation_error(t_est, t_gt):
<<<<<<< HEAD
    assert (t_est.size == t_gt.size == 3)
=======
>>>>>>> 25b9773556d5d7b6584d15ca1687855e187e5716
    t_error = np.linalg.norm(t_gt - t_est)/np.linalg.norm(t_est)
    return t_error

def rotation_error(r_est, r_gt):
<<<<<<< HEAD
    assert (r_est.size == r_gt.size == 3)
=======
>>>>>>> 25b9773556d5d7b6584d15ca1687855e187e5716
    tmp_est = R.from_matrix(r_est)
    tmp_gt = R.from_matrix(r_gt)
    q_est = tmp_est.as_matrix()
    q_gt = tmp_gt.as_matrix()
    r_error = np.linalg.norm(q_gt - q_est)/np.linalg.norm(q_est)
    return r_error

def compute_add_score(pts3d, diameter, R_pred, t_pred, R_gt, t_gt, percentage=0.1):
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]        
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)            

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score
