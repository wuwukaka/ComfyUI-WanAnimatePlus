import cv2
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees:float =55 ):   # nlf default fov_degrees 55
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    # intrinsic_matrix 3*3
    return np.array([   
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1],
    ])


def p3d_to_p2d(point_3d, height, width):    # point3d n*1024*3
    camera_matrix = intrinsic_matrix_from_field_of_view((height,width))
    camera_matrix = np.expand_dims(camera_matrix, axis=0)
    camera_matrix = np.expand_dims(camera_matrix, axis=0)    # 1*1*3*3
    point_3d = np.expand_dims(point_3d,axis=-1)     # n*1024*3*1
    point_2d = (camera_matrix@point_3d).squeeze(-1)
    point_2d[:,:,:2] = point_2d[:,:,:2]/point_2d[:,:,2:3]
    return point_2d[:,:,:]      # n*1024*2


def get_pose_images(smpl_data, offset):
    pose_images = []
    for data in smpl_data:
        if isinstance(data, np.ndarray):
            joints3d = data
        else:
            joints3d = data.numpy()
        canvas = np.zeros(shape=(offset[0], offset[1], 3), dtype=np.uint8)
        joints3d = p3d_to_p2d(joints3d, offset[0], offset[1])
        canvas = draw_3d_points(canvas, joints3d[0], stickwidth=int(offset[1]/350))
        pose_images.append(Image.fromarray(canvas))
    return pose_images


def get_control_conditions(poses, h, w, stick_width=1.0, point_radius=2, style="original"):
    control_images = []
    for idx, pose in enumerate(poses):
        canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        try:
            joints3d = p3d_to_p2d(pose, h, w)
            if style == "original":
                canvas = draw_3d_points(
                    canvas,
                    joints3d[0],
                    stickwidth=int(h / 350 * stick_width),
                    r=point_radius,
                )
            elif style == "scail":
                canvas = draw_3d_points_scail(
                    canvas,
                    joints3d[0],
                    stickwidth=int(h / 350 * stick_width),
                    r=point_radius,
                )
            resized_canvas = cv2.resize(canvas, (w, h))
            # Image.fromarray(resized_canvas).save(f'tmp/{idx}_pose.jpg')
            control_images.append(resized_canvas)
        except Exception:
            control_images.append(Image.fromarray(canvas))
    control_pixel_values = np.array(control_images)
    control_pixel_values = torch.from_numpy(control_pixel_values).contiguous() / 255.
    return control_pixel_values


def draw_3d_points(canvas, points, stickwidth=2, r=2, draw_line=True):
    colors = [
        [255, 0, 0],    # 0
        [0, 255, 0],    # 1
        [0, 0, 255],    # 2
        [255, 0, 255],  # 3
        [255, 255, 0],  # 4
        [85, 255, 0],   # 5
        [0, 75, 255],   # 6
        [0, 255, 85],   # 7
        [0, 255, 170],  # 8
        [170, 0, 255],  # 9
        [85, 0, 255],   # 10
        [0, 85, 255],   # 11
        [0, 255, 255],  # 12
        [85, 0, 255],   # 13
        [170, 0, 255],  # 14
        [255, 0, 255],  # 15
        [255, 0, 170],  # 16
        [255, 0, 85],   # 17
    ]
    connetions = [
        [15,12],[12, 16],[16, 18],[18, 20],[20, 22],
        [12,17],[17,19],[19,21],
        [21,23],[12,9],[9,6],
        [6,3],[3,0],[0,1],
        [1,4],[4,7],[7,10],[0,2],[2,5],[5,8],[8,11]
    ]
    connection_colors = [
        [255, 0, 0],    # 0
        [0, 255, 0],    # 1
        [0, 0, 255],    # 2
        [255, 255, 0],  # 3
        [255, 0, 255],  # 4
        [0, 255, 0],    # 5
        [0, 85, 255],   # 6
        [255, 175, 0],  # 7
        [0, 0, 255],    # 8
        [255, 85, 0],   # 9
        [0, 255, 85],   # 10
        [255, 0, 255],  # 11
        [255, 0, 0],    # 12
        [0, 175, 255],  # 13
        [255, 255, 0],  # 14
        [0, 0, 255],    # 15
        [0, 255, 0],    # 16
    ]

    # draw point
    for i in range(len(points)):
        x,y = points[i][0:2]
        x,y = int(x),int(y)
        if i==13 or i == 14:
            continue
        cv2.circle(canvas, (x, y), r, colors[i%17], thickness=-1)

    # draw line
    if draw_line:
        for i in range(len(connetions)):
            point1_idx,point2_idx = connetions[i][0:2]
            point1 = points[point1_idx]
            point2 = points[point2_idx]
            Y = [point2[0],point1[0]]
            X = [point2[1],point1[1]]
            mX = int(np.mean(X))
            mY = int(np.mean(Y))
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, connection_colors[i%17])

    return canvas

def draw_3d_points_scail(canvas, points, stickwidth=2, r=2, draw_line=True):

    connetions = [
        [15,12],[12, 16],[16, 18],[18, 20],[20, 22],  # 0-4: Left arm chain
        [12,17],[17,19],[19,21],                       # 5-7: Right arm chain
        [21,23],                                       # 8: Right hand
        [12,1],[1,4],[4,7],                           # 9-11: Neck to left leg (hip, thigh, shin)
        [12,2],[2,5],[5,8],                           # 12-14: Neck to right leg (hip, thigh, shin)
    ]

    # Warm colors for right side, cool colors for left side
    connection_colors = [
        [180, 180, 180],    # 0: [15,12] - L. clavicle (Bright Cyan)
        [0, 200, 255],    # 1: [12,16] - L. shoulder (Bright Cyan)
        [0, 120, 255],    # 2: [16,18] - L. upper arm (Bright Blue)
        [0, 60, 255],     # 3: [18,20] - L. forearm (Deep Blue)
        [60, 0, 255],     # 4: [20,22] - L. hand (Blue-Purple)
        [255, 0, 0],      # 5: [12,17] - R. clavicle (Bright Red)
        [255, 100, 0],    # 6: [17,19] - R. upper arm (Bright Orange)
        [255, 180, 0],    # 7: [19,21] - R. forearm (Golden Orange)
        [255, 255, 0],    # 8: [21,23] - R. hand (Bright Yellow)
        [30, 27, 160],    # 9: [12,1] - Neck to L. hip (purple-blue)
        [73, 27, 177],    # 10: [1,4] - L. thigh (purple)
        [145, 27, 194],    # 11: [4,7] - L. shin (magenta)
        [200, 255, 100],    # 12: [12,2] - Neck to R. hip (yellow)
        [54, 201, 52],     # 13: [2,5] - R. thigh (green)
        [30, 176, 85],     # 14: [5,8] - R. shin (green)
    ]

    # draw line
    if draw_line:
        # Collect all joints that are part of connections
        joints_in_use = set()
        for connection in connetions:
            joints_in_use.add(connection[0])
            joints_in_use.add(connection[1])

        for i in range(len(connetions)):
            point1_idx, point2_idx = connetions[i][0:2]
            point1 = points[point1_idx]
            point2 = points[point2_idx]
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            cv2.line(canvas, (x1, y1), (x2, y2), connection_colors[i], stickwidth)

    # draw points for joints that have connections
    joints_in_use = set()
    for connection in connetions:
        joints_in_use.add(connection[0])
        joints_in_use.add(connection[1])

    for joint_idx in joints_in_use:
        if joint_idx >= len(points):
            continue
        x, y = points[joint_idx][0:2]
        x, y = int(x), int(y)
        # Use the color from the first connection involving this joint
        joint_color = [180, 180, 180]  # default grey
        for i, connection in enumerate(connetions):
            if connection[0] == joint_idx or connection[1] == joint_idx:
                joint_color = connection_colors[i]
                break
        cv2.circle(canvas, (x, y), r, joint_color, thickness=-1)

    return canvas
