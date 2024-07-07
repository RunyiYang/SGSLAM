import torch
from torch import nn
import time 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import differential_evolution

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask, l1_loss, l1_loss_calculate


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
            
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, rgb_boundary_threshold, depth_anything, DEVICE, transform, config, render_pkg_input):
        gt_color, gt_depth, gt_pose, raw_image = dataset[idx]
        #np.save(f'/home/wenxuan/MonoGS/tum_debug_images/pose_gt/combined_{idx}', gt_pose.detach().cpu().numpy())
        def depth_anything_depth(image, depth_gt, cur_frame_idx, config, render_pkg_input):
            depth_gt = depth_gt
            predicted_depth = np.zeros_like(depth_gt)
            non_zero_mask = depth_gt != 0
            predicted_depth[non_zero_mask] = 1 / depth_gt[non_zero_mask]
            depth_gt_disparity = predicted_depth
            print('depth_gt_median', np.median(depth_gt))
            #depth_render = depth_gt
            #if render_pkg_input != 0:
            #    depth_render = render_pkg_input["depth"].detach().cpu().numpy()[0]
            #print('render depth', np.median(depth_render))
            time1 = time.time()
            with torch.no_grad():
                depth = depth_anything.infer_image(image, 518)
            time2 = time.time()
            #print('depth_anything time', time2 - time1)
            sigma_color=150
            sigma_space=150
            output = depth.squeeze()
            depthmap = cv2.bilateralFilter(output, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            bounds = [(0,100000), (-5000, 10000)]

            # Use Differential Evolution to optimize the scale and translation
            result = differential_evolution(lambda params: l1_loss(params, depthmap, depth_gt_disparity)[0], bounds)

            # Check the results
            optimal_scale, optimal_translation = result.x
            depth = 1 / (depthmap * optimal_scale + optimal_translation)
            print('median predicted depth', np.median(depth))
            optimal_l1_loss, outlier_mask = l1_loss_calculate(optimal_scale, optimal_translation, depth, depth_gt)
            print('optimal_l1_loss', optimal_l1_loss)
            depth = depth * (1 - outlier_mask)
            return depth
        
        depth_anything_depth_output = depth_anything_depth(raw_image, gt_depth, idx, config, render_pkg_input)
        return Camera(
            idx,
            gt_color,
            depth_anything_depth_output,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
