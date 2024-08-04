import torch
from gaussian_splatting.gaussian_renderer import render
import cv2
from scipy.optimize import differential_evolution
import numpy as np
import os

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    #if config["Training"]["monocular"]:
    #    return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    #if config["Training"]["monocular"]:
    #    return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()
    # print("gt_image", gt_image.shape)


    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    # print("gt_depth", gt_depth.shape)
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def prune_list(gaussians, viewpoint_stack, pipeline_params, background):
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    # ic(dataset.model_path)
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
    return gaussian_list, imp_list

def prune_list_global(gaussians, viewpoint_stack, pipeline_params, background):
    gaussian_list, imp_list = None, None
    print('viewpoint_stack', viewpoint_stack)
    viewpoint_cam = viewpoint_stack.popitem()
    render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    # ic(dataset.model_path)
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
    return gaussian_list, imp_list

def prune_gaussians(percent, import_score):
    #print('imporatant score', import_score.shape)
    sorted_tensor, _ = torch.sort(import_score, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (import_score <= value_nth_percentile).squeeze()
    pruned_import_score = import_score[~prune_mask]
    return prune_mask, pruned_import_score

def prune_gaussians_threshold(threshold, import_score):
    """
    Prunes elements of import_score based on a threshold value.

    Parameters:
        threshold (float): The cutoff value below which elements are pruned.
        import_score (torch.Tensor): The tensor of scores to be pruned.

    Returns:
        torch.Tensor: A mask indicating which elements were pruned (True if pruned).
        torch.Tensor: The pruned tensor with elements above the threshold.
    """
    # Create a boolean mask where True values indicate scores <= threshold
    prune_mask = (import_score <= threshold).squeeze()
    
    # Use the mask to select elements that are above the threshold
    pruned_import_score = import_score[~prune_mask]

    return prune_mask

def disparity_loss(depth_da):
    #depth_gt = depth_render
    sigma_color=150
    sigma_space=150
    output = depth_da.squeeze()
    depthmap = cv2.bilateralFilter(output, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Use Differential Evolution to optimize the scale and translation
    depth = depthmap
    return depth, 0

def save_depth_image_with_viridis(depth_image, output_path):
    # Normalize the depth image to fit within the range 0-255 as uint8
    depth_normalized = cv2.normalize(depth_image * 5000.0, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Apply the viridis colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

    # Save the colored depth image
    cv2.imwrite(output_path, depth_colored)

def calculate_scale(depth_render, depth_da, idx):
    sigma_color=150
    sigma_space=150
    depth_gt = depth_render
    output = depth_gt.squeeze().astype(np.float32)
    depth_gt = cv2.bilateralFilter(output, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    #print('depth_gt_median', np.median(depth_gt))
    depth = depth_da
    output = depth.squeeze()
    depthmap = cv2.bilateralFilter(output, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    bounds = [(-10,50), (0, 10)]

    # Use Differential Evolution to optimize the scale and translation
    result = differential_evolution(lambda params: l1_loss(params, depthmap, depth_gt)[0], bounds)

    # Check the results
    optimal_scale, optimal_translation = result.x
    print('optimal_scale', optimal_scale)
    print('optimal_translation', optimal_translation)
    depth = depthmap * optimal_scale + optimal_translation
    print('median predicted depth', np.median(depth))
    optimal_l1_loss, outlier_mask = l1_loss_calculate(optimal_scale, optimal_translation, depth, depth_gt)
    print('optimal_l1_loss_before_outlier', optimal_l1_loss)
    
    # Define the directories
    scale_dir = '/home/wenxuan/MonoGS/tum_debug_images/scale_metric_desk'
    translation_dir = '/home/wenxuan/MonoGS/tum_debug_images/translation_metric_desk'
    loss_dir = '/home/wenxuan/MonoGS/tum_debug_images/loss_metric_desk'

    # Create the directories if they do not exist
    os.makedirs(scale_dir, exist_ok=True)
    os.makedirs(translation_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
                
    # Save the data to .npy files
    np.save(f'{scale_dir}/combined_{idx}', optimal_scale)
    np.save(f'{translation_dir}/combined_{idx}', optimal_translation)
    np.save(f'{loss_dir}/combined_{idx}', optimal_l1_loss)

def l1_loss(params, predicted_depth, depth_gt, center_fraction=1.0):
    """
    Calculate the L1 loss using a central portion of the depth ground truth.
    
    Args:
    params (tuple): Scale and translation parameters.
    predicted_depth (np.array): Predicted depth map.
    depth_gt (np.array): Ground truth depth map.
    center_fraction (float): Fraction of the central area of the depth map to consider (0 < center_fraction <= 1).

    Returns:
    float: Mean of absolute differences in the specified central area.
    np.array: Outlier mask (not implemented in detail here).
    """
    scale, translation = params
    scaled_and_translated_predicted = predicted_depth * scale + translation

    # Validate center_fraction
    if not (0 < center_fraction <= 1):
        raise ValueError("center_fraction must be between 0 and 1.")

    # Calculate the center region dimensions based on center_fraction
    h, w = depth_gt.shape
    reduce_h = int(h * (1 - center_fraction) / 2)
    reduce_w = int(w * (1 - center_fraction) / 2)

    # Create a mask where GT depth values are not zero and within the specified center fraction
    mask = np.zeros_like(depth_gt, dtype=bool)
    mask[reduce_h:h-reduce_h, reduce_w:w-reduce_w] = True
    mask = mask & (depth_gt != 0)

    # Apply the mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[mask]
    valid_predicted_depth = scaled_and_translated_predicted[mask]

    # Calculate the mean of absolute differences where GT depth is not zero
    loss_map = np.abs(valid_gt_depth - valid_predicted_depth)

    # Create an outlier mask (optional, depending on additional requirements)
    outlier_mask = np.zeros_like(depth_gt, dtype=bool)  # Implementation needed if outliers are to be detected

    # Return the mean of absolute differences and the outlier mask
    return np.mean(loss_map), outlier_mask

def l1_loss_partial(params, predicted_depth, depth_gt):
    """
    Calculate the L1 loss using regions of the depth ground truth within one standard deviation of the median depth.
    
    Args:
    params (tuple): Scale and translation parameters.
    predicted_depth (np.array): Predicted depth map.
    depth_gt (np.array): Ground truth depth map.

    Returns:
    float: Mean of absolute differences in the specified area.
    np.array: Outlier mask (not implemented in detail here).
    """
    scale, translation = params
    scaled_and_translated_predicted = predicted_depth * scale + translation

    # Mask where GT depth values are not zero
    mask = depth_gt != 0

    # Calculate the median and standard deviation of the valid (non-zero) ground truth depths
    valid_gt_depth = depth_gt[mask]
    median_depth = np.median(valid_gt_depth)
    std_depth = np.std(valid_gt_depth)

    # Create a mask to only include depths within one standard deviation of the median depth
    std_mask = np.abs(depth_gt - median_depth) <= std_depth
    #print(median_depth)
    #print(std_depth)
    #print('left', median_depth + std_depth)
    #print('right', median_depth - std_depth)
    # Combine the non-zero mask and the standard deviation mask
    final_mask = mask & std_mask

    # Apply the final mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[final_mask]
    valid_predicted_depth = scaled_and_translated_predicted[final_mask]

    # Calculate the mean of absolute differences where GT depth is not zero
    loss_map = np.abs(valid_gt_depth - valid_predicted_depth)

    # Create an outlier mask (optional, depending on additional requirements)
    outlier_mask = np.zeros_like(depth_gt, dtype=bool)  # Implementation needed if outliers are to be detected

    # Return the mean of absolute differences and the outlier mask
    return np.mean(loss_map), outlier_mask

def l1_loss_calculate(scale, translation, predicted_depth, depth_gt, percentile_threshold = 80):
    # Apply scale and translation to predicted depth
    
    # Create a mask where GT depth values are not zero
    mask_non_zero = depth_gt != 0
    
    # Apply the mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[mask_non_zero]
    valid_predicted_depth = predicted_depth[mask_non_zero]
    #print('Valid depth shape:', valid_predicted_depth.shape)
    
    # Calculate the L1 loss where GT depth is not zero
    loss_map = np.abs(valid_gt_depth - valid_predicted_depth)
    #print('Loss map shape:', loss_map.shape)
    #print('mean loss_map', np.mean(loss_map))
    
    # Calculate the threshold for outliers based on the specified percentile of the loss map
    threshold = np.percentile(loss_map, percentile_threshold)
    #print('Outlier threshold:', threshold)
    
    # Identify outliers within the non-zero mask
    outliers_in_non_zero = loss_map > threshold
    
    # Create the full outlier mask, initializing with False for every element
    outlier_mask = np.zeros_like(depth_gt, dtype=bool)
    # Only mark as outliers those that are non-zero and exceed the loss threshold
    outlier_mask[mask_non_zero] = outliers_in_non_zero
    
    # Calculate the number and percentage of values masked out as outliers
    num_masked_outliers = np.sum(outliers_in_non_zero)
    total_values = np.sum(mask_non_zero)  # total non-zero values
    percent_masked_outliers = (num_masked_outliers / total_values) * 100 if total_values != 0 else 0
    
    #print(f"Number of masked outliers: {num_masked_outliers}")
    #print(f"Percentage of masked outliers: {percent_masked_outliers:.2f}%")
    valid_loss_map = loss_map[~outliers_in_non_zero]  # Exclude outliers for mean calculation
    return np.mean(loss_map), outlier_mask

