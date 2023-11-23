import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from sklearn.preprocessing import MinMaxScaler
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2

pose_landmark_idxs = [11, 12]


def algorithm_a(hand_results, pose_results) -> np.ndarray:
    """
    相对位置最小最大缩放,维度为 2 * （21 + 2） * 3 + 即 138维
    0-63 维存储 左手向量
    63-126 维存储 右手向量
    126-138 维存储 肩膀向量
    138-142 旋转矩阵
    """
    normalized_landmarks = np.zeros(142)
    hand_landmarks_list = hand_results.hand_landmarks
    pose_landmarks_list = pose_results.pose_landmarks
    handedness_list = hand_results.handedness
    left_hand_center = [0]*3
    right_hand_center = [0]*3
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])
        hand_center = np.array(hand_landmarks[0])
        x_min_landmark = min(hand_landmarks[:, 0])
        hand_landmarks[:, 0] -= x_min_landmark
        y_min_landmark = min(hand_landmarks[:, 1])
        hand_landmarks[:, 1] -= y_min_landmark
        z_min_landmark = min(hand_landmarks[:, 2])
        hand_landmarks[:, 2] -= z_min_landmark
        for idx2 in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx2]
            dot_A = np.array([pose_landmarks[11].x, pose_landmarks[11].y])
            dot_B = np.array([pose_landmarks[12].x, pose_landmarks[12].y])
            dot_C = np.array([hand_landmarks[0][0],hand_landmarks[0][1]])
            doc_D = np.array([hand_landmarks[9][0],hand_landmarks[9][1]])
            AB, CD = dot_B - dot_A, doc_D - dot_C
            # 计算垂直向量
            AB_complementary = np.array([-AB[1], AB[0]])
            CD_complementary = np.array([-CD[1], CD[0]])
            # 归一化垂直向量（单位向量）
            AB_complementary = AB_complementary / np.linalg.norm(AB_complementary)
            CD_complementary = CD_complementary / np.linalg.norm(CD_complementary)
            # 计算旋转矩阵
            rotation_matrix = np.array([[np.dot(AB_complementary, CD_complementary),
                                         -np.sqrt(1 - np.dot(AB_complementary, CD_complementary) ** 2)],
                                        [np.sqrt(1 - np.dot(AB_complementary, CD_complementary) ** 2),
                                         np.dot(AB_complementary, CD_complementary)]])
            normalized_landmarks[138:142] = scaler.fit_transform(rotation_matrix.reshape(-1,1)).flatten()
            for i in range(21):
                XY = hand_landmarks[i,0:2]-dot_C
                hand_landmarks[i, 0:2] = np.dot(rotation_matrix, XY) + dot_C
        normalized_hand_landmarks = scaler.fit_transform(hand_landmarks.reshape(-1, 1)).flatten()
        if handedness[0].category_name == "Left":
            normalized_landmarks[0:63] = normalized_hand_landmarks
            left_hand_center = hand_center.flatten()
        else:
            normalized_landmarks[63:126] = normalized_hand_landmarks
            right_hand_center = hand_center.flatten()
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks = [[pose_landmarks[idx2].x,
                            pose_landmarks[idx2].y,
                            pose_landmarks[idx2].z] for idx2 in pose_landmark_idxs]
        pose_landmarks.extend([right_hand_center,left_hand_center])
        pose_landmarks = np.array(pose_landmarks)
        x_min_landmark = min(pose_landmarks[:, 0])
        pose_landmarks[:, 0] -= x_min_landmark
        y_min_landmark = min(pose_landmarks[:, 1])
        pose_landmarks[:, 1] -= y_min_landmark
        z_min_landmark = min(pose_landmarks[:, 2])
        pose_landmarks[:, 2] -= z_min_landmark
        normalized_pose_landmarks = scaler.fit_transform(pose_landmarks.reshape(-1, 1)).flatten()
        normalized_landmarks[126:138] = normalized_pose_landmarks
    return normalized_landmarks


# 创建 MinMaxScaler 对象
scaler = MinMaxScaler(feature_range=(0, 1))


def algorithm_b(hand_results: HandLandmarkerResult, pose_results: PoseLandmarkerResult) -> np.ndarray:
    """
    相对位置最小最大缩放,维度为 2 * （21 + 2） * 3 即 138维
    0-63 维存储 左手向量
    63-126 维存储 右手向量
    126-138 维存储 肩膀向量
    """
    normalized_landmarks = np.zeros(138)
    hand_landmarks_list = hand_results.hand_landmarks
    pose_landmarks_list = pose_results.pose_landmarks
    handedness_list = hand_results.handedness
    left_hand_center = [0]*3
    right_hand_center = [0]*3
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])
        hand_center = np.array(hand_landmarks[0])
        x_min_landmark = min(hand_landmarks[:, 0])
        hand_landmarks[:, 0] -= x_min_landmark
        y_min_landmark = min(hand_landmarks[:, 1])
        hand_landmarks[:, 1] -= y_min_landmark
        z_min_landmark = min(hand_landmarks[:, 2])
        hand_landmarks[:, 2] -= z_min_landmark
        normalized_hand_landmarks = scaler.fit_transform(hand_landmarks.reshape(-1, 1)).flatten()
        if handedness[0].category_name == "Left":
            normalized_landmarks[0:63] = normalized_hand_landmarks
            left_hand_center = hand_center.flatten()
        else:
            normalized_landmarks[63:126] = normalized_hand_landmarks
            right_hand_center = hand_center.flatten()
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks = [[pose_landmarks[idx2].x,
                            pose_landmarks[idx2].y,
                            pose_landmarks[idx2].z] for idx2 in pose_landmark_idxs]
        pose_landmarks.extend([right_hand_center,left_hand_center])
        pose_landmarks = np.array(pose_landmarks)
        x_min_landmark = min(pose_landmarks[:, 0])
        pose_landmarks[:, 0] -= x_min_landmark
        y_min_landmark = min(pose_landmarks[:, 1])
        pose_landmarks[:, 1] -= y_min_landmark
        z_min_landmark = min(pose_landmarks[:, 2])
        pose_landmarks[:, 2] -= z_min_landmark
        normalized_pose_landmarks = scaler.fit_transform(pose_landmarks.reshape(-1, 1)).flatten()
        normalized_landmarks[126:138] = normalized_pose_landmarks
    return normalized_landmarks


def calculate(hand_results, pose_results, operation) -> np.ndarray:
    return operation(hand_results, pose_results)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, hand_results, pose_results):
    hand_landmarks_list = hand_results.hand_landmarks
    pose_landmarks_list = pose_results.pose_landmarks

    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for idx2 in pose_landmark_idxs:
            pose_landmarks_proto.landmark.append(
                landmark_pb2.NormalizedLandmark(x=pose_landmarks[idx2].x, y=pose_landmarks[idx2].y,
                                                z=pose_landmarks[idx2].z)
            )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto)

    return annotated_image
