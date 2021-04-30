import os, sys
import cv2
from cv2 import aruco

TARGET_ARUCO_DICT_NUM = aruco.DICT_5X5_250

locate_dir = sys.path[0] + '/'
out_dir_path = locate_dir + 'data/calibration/'

camera_matrix_tag_name = 'camera_matrix'
camera_matrix_file_name = camera_matrix_tag_name + '.xml'
camera_matrix_path = out_dir_path + camera_matrix_file_name

dist_coeffs_tag_name = 'distortion_coefficients'
dist_coeffs_file_name = dist_coeffs_tag_name + '.xml'
dist_coeffs_path = out_dir_path + dist_coeffs_file_name

def detect_aruco_marker(input_img):
    #print('Detect ArUco marker')
    gray_input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    target_aruco_dict = aruco.Dictionary_get(TARGET_ARUCO_DICT_NUM)
    detect_params =  aruco.DetectorParameters_create()
    marker_corners, marker_ids, rejected_img_points = aruco.detectMarkers(
            gray_input_img, target_aruco_dict, parameters=detect_params)

    marker_num = len(marker_corners)
    reject_marker_num = len(rejected_img_points)
    if len(marker_corners) > 0:
        #print('=> Detect {0} ArUco marker(s)'.format(marker_num))
        # sub pixel detection
        sub_pix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        for corner in marker_corners:
            cv2.cornerSubPix(
                    gray_input_img,
                    corner,
                    winSize = (3, 3),
                    zeroZone = (-1, -1),
                    criteria = sub_pix_criteria)
    #elif reject_marker_num > 0:
    #    print('=> Reject {0} marker(s)'.format(reject_marker_num))
    #else:
    #    print('=> No ArUco marker is detected')

    return marker_num, marker_corners, marker_ids

def mark_3d_points(input_img, object_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    #print('Projects 3D points to input image')
    proj_points, jac_mat = cv2.projectPoints(
            object_points, rotation_vector, translation_vector,
            camera_matrix, dist_coeffs)
    for i in range(len(proj_points)):
        img_pt_x = int(proj_points[i][0][0])
        img_pt_y = int(proj_points[i][0][1])
        msg = '=> (%.2f, %.2f, %.2f) -> (%d, %d)' % (object_points[i][0], object_points[i][1], object_points[i][2], img_pt_x, img_pt_y)
        #print(msg)
        cv2.circle(input_img, (img_pt_x, img_pt_y), 5, (255, 0, 0), -1)
    return;

def draw_axis(input_img, marker_corners, marker_len, axis_len, camera_matrix, dist_coeffs):
    #print('Draw axis to ArUco marker (pose estimation)')
    rotation_vectors, translation_vectors, obj_points = aruco.estimatePoseSingleMarkers(
            marker_corners, marker_len, camera_matrix, dist_coeffs)
    if rotation_vectors is not None:
        img_axis = input_img.copy()
        for i in range(len(rotation_vectors)):
            img_axis = aruco.drawAxis(
                    img_axis, camera_matrix, dist_coeffs,
                    rotation_vectors[i], translation_vectors[i], axis_len)
    else:
        img_axis = None
        print('Error: cannot estimate pose')

    return img_axis, rotation_vectors, translation_vectors

def calibration_data_load():
    print('Load camera calibration data')
    if not os.path.isfile(camera_matrix_path):
        msg = 'Error: cannot found %s' % camera_matrix_path
        print(msg)
        sys.exit(1)

    if not os.path.isfile(dist_coeffs_path):
        msg = 'Error: cannot found %s' % dist_coeffs_path
        print(msg)
        sys.exit(1)

    cv_fs = cv2.FileStorage(camera_matrix_path, cv2.FILE_STORAGE_READ)
    camera_matrix_node = cv_fs.getNode(camera_matrix_tag_name)
    camera_matrix = camera_matrix_node.mat()
    print('=> Intrinsic matrix:\n{0}'.format(camera_matrix))

    cv_fs = cv2.FileStorage(dist_coeffs_path, cv2.FILE_STORAGE_READ)
    dist_coeffs_node = cv_fs.getNode(dist_coeffs_tag_name)
    dist_coeffs = dist_coeffs_node.mat()
    print('=> Distortion coefficients:\n{0}'.format(dist_coeffs))

    return camera_matrix, dist_coeffs
