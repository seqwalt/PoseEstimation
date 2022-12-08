import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

def translation_error(t_est, t_gt):
    t_error = np.linalg.norm(t_gt - t_est)/np.linalg.norm(t_est)
    return t_error

def rotation_error(r_est, r_gt):
    tmp_est = R.from_matrix(r_est)
    tmp_gt = R.from_matrix(r_gt)
    q_est = tmp_est.as_matrix()
    q_gt = tmp_gt.as_matrix()
    r_error = np.linalg.norm(q_gt - q_est)/np.linalg.norm(q_est)
    return r_error

def compute_add_score(pts3d, diameter, R_pred, t_pred, R_gt, t_gt, percentage=0.1):
    count = pts3d.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt * pts3d[i].transpose() + t_gt.transpose() 
        pts_xformed_pred = R_pred * pts3d[i].transpose() + t_pred.transpose()
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)            

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_adds_score(pts3d, diameter, R_pred, t_pred, R_gt, t_gt, percentage=0.1):
    count = pts3d.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt * pts3d[i].transpose() + t_gt.transpose()
        pts_xformed_pred = R_pred * pts3d[i].transpose() + t_pred.transpose()
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score


#example for translation error 
#gournd truth
t_gt0 = np.array([3, 2, 1])
#estimation
t_est1 = np.array([[3, 2, 1]]) 
t_est2 = np.array([[4, 5, 6]])

#translation error
t_err1 = translation_error(t_est1, t_gt0)
t_err2 = translation_error(t_est2, t_gt0)

print("translation error")
print(t_err1, t_err2)

#example for rotation error
#ground truth
r_gt0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#estmation 
r_est1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
r_est2 = np.array([[1, 0, 0], [ 0, 0, -1],[ 0, 1, 0]])

#rotation error
r_err1 = rotation_error(r_est1, r_gt0)
r_err2 = rotation_error(r_est2, r_gt0)

print("rotation error")
print (r_err1, r_err2)

#Example for ADD(-S)
#sample points
pts3d0 = np.array([[100,100,100],[5,10,10],[1,2,3],[2,2,2]])
#model diameter
diameter0 = 100

add1 = compute_add_score(pts3d0, diameter0, r_est1, t_est1, r_gt0, t_gt0)
adds1 = compute_adds_score(pts3d0, diameter0, r_est1, t_est1, r_gt0, t_gt0)

add2 = compute_add_score(pts3d0, diameter0, r_est2, t_est2, r_gt0, t_gt0)
adds2 = compute_adds_score(pts3d0, diameter0, r_est2, t_est2, r_gt0, t_gt0)

print("ADD & ADD-s")
print (add1, adds1, add2, adds2)