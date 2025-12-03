#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from typing import List
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import cv2
import pdb
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    # pdb.set_trace()
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        print(f"resizing  from {orig_w} to {orig_h} to {resolution}")

    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    # print(np.unique(cam_info.mask))
    # Prepare mask if available (similar to foreground reconstruction)
    if cam_info.mask is not None:
        print(f"resizing to {resolution}")
        print("cam_info.mask.shape", cam_info.mask.shape)
        print(np.unique(cam_info.mask))
        mask = cv2.resize(cam_info.mask, resolution, interpolation=cv2.INTER_NEAREST)   # NOTE: Mask needs to be downscaled as well 
        print(np.unique(mask), mask.shape, "post resize")
        mask = mask/255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
    else:
        print("no mask")
        # pdb.set_trace()
        mask = None

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    alpha_mask = None

    if resized_image_rgb.shape[0] == 4:
        alpha_mask = resized_image_rgb[3, ...]

    # Rescale intrinsics if downsizing
    fx = cam_info.fx * 1/args.resolution
    fy = cam_info.fy * 1/args.resolution
    cx = cam_info.cx * 1/args.resolution
    cy = cam_info.cy * 1/args.resolution

    FovY = focal2fov(fy, resolution[1])
    FovX = focal2fov(fx, resolution[0])
    cx_ratio=2*cx/resolution[0]
    cy_ratio=2*cy/resolution[1]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=FovX, FoVy=FovY, 
                  image=gt_image, gt_alpha_mask=alpha_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  mask=mask, cx_ratio=cx_ratio, cy_ratio=cy_ratio, 
                  cx=cx, cy=cy)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def get_cameras_spatial_extent(cameras:List[Camera]):
    cam_centers = torch.cat([camera.camera_center.view(1, 3) for camera in cameras], dim=0)

    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
    dist = torch.norm(cam_centers - avg_cam_center, dim=1, keepdim=True)

    half_diagonal = torch.max(dist)
    radius = half_diagonal * 1.1

    translate = -avg_cam_center

    return {"translate": translate, "radius": radius, "avg_cam_center": avg_cam_center}


# def rescale_intrinsics(intrinsics, scale):
#     intrinsics[0, 0] = intrinsics[0, 0] * 1/scale # fx 
#     intrinsics[1, 1] = intrinsics[1, 1] * 1/scale # fy
#     intrinsics[0, 2] = intrinsics[0, 2] * 1/scale # cx
#     intrinsics[1, 2] = intrinsics[1, 2] * 1/scale # cy
#     return intrinsics