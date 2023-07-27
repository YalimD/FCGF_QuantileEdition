import copy

import open3d as o3d
import os
import logging
import numpy as np

from util.trajectory import CameraPose
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy


def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
    distance_threshold = voxel_size * 1.5
    # Set mutual filter to False.
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        xyz0, xyz1, feat0, feat1, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result_ransac.transformation


def gather_results(results):
    traj = []
    for r in results:
        success = r[0]
        if success:
            traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
    return traj


def gen_matching_pair(pts_num):
    matching_pairs = []
    for i in range(pts_num):
        for j in range(i + 1, pts_num):
            matching_pairs.append([i, j, pts_num])
    return matching_pairs


def read_data(feature_path, name):
    data = np.load(os.path.join(feature_path, name + ".npz"))
    xyz = make_open3d_point_cloud(data['xyz'])
    feat = make_open3d_feature_from_numpy(data['feature'])
    return data['points'], xyz, feat


def run_quantile(xyz_i, xyz_j, feat_i, feat_j, voxel_size, target_size, matching_method, tuple_test, alpha):
    import quantile_assignment

    feat_i_c = copy.deepcopy(feat_i).data
    feat_j_c = copy.deepcopy(feat_j).data

    xyz_i_c = copy.deepcopy(xyz_i)
    xyz_j_c = copy.deepcopy(xyz_j)

    data_indices = np.linspace(0, feat_i_c.shape[1] - 1, target_size, dtype="int")

    # Sample a subset of the features
    feat_i_c = feat_i_c[:, data_indices]
    feat_j_c = feat_j_c[:, data_indices]

    xyz_i_c = xyz_i_c.select_by_index(data_indices)
    xyz_j_c = xyz_j_c.select_by_index(data_indices)

    result = quantile_assignment.quantile_registration(feat_i_c.T,
                                                       feat_j_c.T,
                                                       1.0,
                                                       False,
                                                       alpha,
                                                       quantile_assignment.matching_lib[matching_method])

    matches, weights, alpha, k_alpha, best_q, cost = result

    matches_arr = np.asarray(matches)
    matches_arr = matches_arr.T
    corr = o3d.utility.Vector2iVector(matches_arr)

    result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(xyz_i_c, xyz_j_c, corr,
                                                                                 o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                     tuple_test=tuple_test))
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_icp(
        xyz_i, xyz_j, distance_threshold, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False))

    return result.transformation


def do_single_pair_matching(parameters):
    feature_path, set_name, m, voxel_size, target_size, matching_method, tuple_test, alpha = parameters

    i, j, s = m
    name_i = "%s_%03d" % (set_name, i)
    name_j = "%s_%03d" % (set_name, j)
    logging.info("matching %s %s" % (name_i, name_j))
    points_i, xyz_i, feat_i = read_data(feature_path, name_i)
    points_j, xyz_j, feat_j = read_data(feature_path, name_j)

    # TODO: This shouldnt matter?
    if len(xyz_i.points) < len(xyz_j.points):
        trans = run_quantile(xyz_i, xyz_j, feat_i, feat_j, voxel_size, target_size, matching_method, tuple_test, alpha)
    else:
        trans = run_quantile(xyz_j, xyz_i, feat_j, feat_i, voxel_size, target_size, matching_method, tuple_test, alpha)
        trans = np.linalg.inv(trans)
    ratio = compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
    logging.info(f"{ratio}")
    if ratio > 0.3:
        return [True, i, j, s, np.linalg.inv(trans)]
    else:
        return [False, i, j, s, np.identity(4)]
