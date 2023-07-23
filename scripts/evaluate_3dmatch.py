import os
import numpy as np
import math

# Papers:
# The data used here is the 8 scenes included in 3DMatch
# The original scenes are from :
# - 7-scenes
# - sun3D
# https://3dmatch.cs.princeton.edu/
# The testing protocol is from:
# http://redwood-data.org/indoor/registration.html


def dataset_format(partition_S, partition_T, num_of_partitions,
                   transformation):
    report = f"{partition_T}\t{partition_S}\t{num_of_partitions}\n"
    for row in range(4):
        report += f"{transformation[row][0]:.8E}\t{transformation[row][1]:.8E}\t" \
                  f"{transformation[row][2]:.8E}\t{transformation[row][3]:.8E}\n"
    return report

# region gt_processing


def process_gt_log(gt_file):
    match_table = None
    accepted_num_of_matches = 0
    total_num_of_matches = 0

    with open(gt_file, "r") as log_reader:
        while log_reader.readable():
            line = log_reader.readline()
            if len(line) == 0:
                break
            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            if match_table is None:
                match_table = [None] * math.ceil(number_of_fragments * (number_of_fragments + 1) / 2)

            transformation = np.zeros((4, 4))
            for r in range(4):
                row = list(map(lambda term: float(term), log_reader.readline().split()))
                transformation[r] = np.asarray(row)

            match_table[int(source * (source + 1) / 2) + target] = transformation

            total_num_of_matches += 1
            if source > target + 1:
                accepted_num_of_matches += 1

    return match_table, accepted_num_of_matches, total_num_of_matches


def process_gt_info(gt_info):
    info_table = None
    accepted_num_of_matches = 0
    total_num_of_matches = 0

    with open(gt_info, "r") as info_reader:
        while info_reader.readable():
            line = info_reader.readline()
            if len(line) == 0:
                break
            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            if info_table is None:
                info_table = [None] * math.ceil(number_of_fragments * (number_of_fragments + 1) / 2)

            info_data = np.zeros((6, 6))
            for r in range(6):
                row = list(map(lambda term: float(term), info_reader.readline().split()))
                info_data[r] = np.asarray(row)

            info_table[int(source * (source + 1) / 2) + target] = info_data

            total_num_of_matches += 1
            if source > target + 1:
                accepted_num_of_matches += 1

    return info_table, accepted_num_of_matches, total_num_of_matches


def convert_to_quaternion(rotation):

    quaternion = np.zeros(4)

    quaternion[0] = 0.5 * np.sqrt(1 + rotation[0][0] + rotation[1][1] + rotation[2][2])
    quaternion[1] = -(rotation[2][1] - rotation[1][2]) / (4 * quaternion[0])
    quaternion[2] = -(rotation[0][2] - rotation[2][0]) / (4 * quaternion[0])
    quaternion[3] = -(rotation[1][0] - rotation[0][1]) / (4 * quaternion[0])

    return quaternion

def calculate_rmse(transformation, gt_trans, info_mat):
    joint = np.linalg.inv(gt_trans) @ transformation

    translation = joint[0:3, 3]
    rotation = joint[:3, :3]

    q_rot = convert_to_quaternion(rotation)
    rho = np.asarray([*translation, *(-q_rot[1:])])

    return (rho.T @ info_mat @ rho) / info_mat[0][0]


def evaluate_fragment_registration(results_file_name, gt_folder, rmse_distance_threshold=0.2):
    gt_file = os.path.join(gt_folder, "gt.log")
    gt_info = os.path.join(gt_folder, "gt.info")

    log_table, accepted_number_of_matches, _ = process_gt_log(gt_file)
    info_table, *_ = process_gt_info(gt_info)

    good_matches = 0
    total_matches = 0

    # Find the number of correct registrations from gt. The indices must be non-consecutive, ignore any such
    with open(results_file_name, "r") as result_reader:
        while result_reader.readable():
            line = result_reader.readline()
            if len(line) == 0:
                break

            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            # Only consider the non-consecutive matches
            if source > target + 1:
                total_matches += 1

                # If it is also a match in gt
                current_index = int(source * (source + 1) / 2) + target
                gt_trans = log_table[current_index]
                info_mat = info_table[current_index]

                if gt_trans is not None and info_mat is not None:

                    transformation = np.zeros((4, 4))
                    for r in range(4):
                        row = list(map(lambda term: float(term), result_reader.readline().split()))
                        transformation[r] = np.asarray(row)

                    rmse_p = calculate_rmse(transformation, gt_trans, info_mat)
                    if rmse_p <= np.power(rmse_distance_threshold, 2):
                        good_matches += 1
                else:  # Skip 4 lines
                    for r in range(4):
                        result_reader.readline()
            else:  # Skip 4 lines
                for r in range(4):
                    result_reader.readline()

    recall = good_matches / accepted_number_of_matches
    precision = good_matches / max(total_matches, 1)

    return recall, precision


#endregion


