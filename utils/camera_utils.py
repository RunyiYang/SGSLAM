import torch
from torch import nn
import time 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import differential_evolution

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask, l1_loss


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
        gt_color, gt_depth, gt_pose = dataset[idx]
        input_depth_anything = gt_color.permute(1, 2, 0).cpu().numpy()
        
        #valid_rgb = (gt_color.sum(dim=0) > rgb_boundary_threshold)[None]
        #gt_depth[~valid_rgb[0].cpu()] = 0  # Ignore the invalid rgb pixels
        
        def depth_anything_depth(image, depth_gt, cur_frame_idx, config, render_pkg_input):
            png_depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
            depth_gt = depth_gt * png_depth_scale
            print('depth_gt', np.max(depth_gt / png_depth_scale))
            if render_pkg_input != 0:
                depth_gt = render_pkg_input["depth"].detach().cpu().numpy()[0] * png_depth_scale
            print('render depth_gt', np.max(depth_gt / 5000.0))
            
            h, w = image.shape[:2]
            image_transform = transform({'image': image})['image']
            image_input = torch.from_numpy(image_transform).unsqueeze(0).to(DEVICE)
            time1 = time.time()
            with torch.no_grad():
                depth = depth_anything(image_input)
            time2 = time.time()
            #print('depth_anything time', time2 - time1)
            
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            sigma_color=150
            sigma_space=150
            output = depth.squeeze().cpu().numpy()
            depthmap = cv2.bilateralFilter(output, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            #optimal_scale, optimal_translation = 13238.90159861579, 3951.3636687612507
            #predicted_depth = np.zeros_like(depthmap)
            #non_zero_mask = depthmap != 0
            predicted_depth = 1  - depthmap
            # Set bounds for scale and translation
            #bounds = [(0,50000), (-5000, 5000)]
            bounds = [(0,100000), (-5000, 10000)]

            # Use Differential Evolution to optimize the scale and translation
            result = differential_evolution(lambda params: l1_loss(params, predicted_depth, depth_gt, png_depth_scale)[0], bounds)

            # Check the results
            optimal_scale, optimal_translation = result.x
            print("Optimal scale:", optimal_scale)
            print("Optimal translation:", optimal_translation)

            # Compute the final depth using optimal scale and translation
            depth = (predicted_depth * optimal_scale + optimal_translation).astype("uint16")
            print('max predicted depth', np.max(depth))
            # Compute the L1 loss at the optimal scale and translation and identify outliers
            optimal_l1_loss, outlier_mask = l1_loss(result.x, predicted_depth, depth_gt, png_depth_scale)
            
            print("L1 loss at optimal scale and translation:", optimal_l1_loss)
            
            '''
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            im_gt = ax[0].imshow(depth_gt, cmap='gray')
            ax[0].set_title('Ground Truth Depth')
            ax[0].axis('off')
            fig.colorbar(im_gt, ax=ax[0], fraction=0.046, pad=0.04)

            im_pred = ax[1].imshow(depth, cmap='gray')
            ax[1].set_title('Predicted Depth')
            ax[1].axis('off')
            fig.colorbar(im_pred, ax=ax[1], fraction=0.046, pad=0.04)

            im_loss = ax[2].imshow(np.abs((depth_gt / png_depth_scale) - (depth / png_depth_scale)), cmap='hot')
            ax[2].set_title(f'L1 Loss: {optimal_l1_loss:.2f}')
            ax[2].axis('off')
            fig.colorbar(im_loss, ax=ax[2], fraction=0.046, pad=0.04)

            # Save the figure
            plt.tight_layout()
            plt.savefig(f'/home/wenxuan/MonoGS/pred_render_depth/combined_{cur_frame_idx}.png')
            plt.close()
            '''
            return depth / png_depth_scale
        
        depth_anything_depth_output = depth_anything_depth(input_depth_anything, gt_depth, idx, config, render_pkg_input)
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
