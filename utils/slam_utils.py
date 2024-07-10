import torch
import numpy as np

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

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
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

def get_median_depth_da(depth, opacity=None, mask=None, return_std=False):
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

def l1_loss(params, predicted_depth, depth_gt):
    scale, translation = params
    scaled_and_translated_predicted = predicted_depth * scale + translation
    
    # Create a mask where GT depth values are not zero
    mask = depth_gt != 0
    
    # Apply the mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[mask]
    valid_predicted_depth = scaled_and_translated_predicted[mask]
    
    # Calculate the mean of absolute differences where GT depth is not zero
    loss_map = np.abs(valid_gt_depth - valid_predicted_depth)
    
    outlier_mask = 0
    
    # Return the mean of absolute differences and the outlier mask
    return np.mean(loss_map), outlier_mask

def l1_loss_calculate(scale, translation, predicted_depth, depth_gt, percentile_threshold = 95):
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

def l2_loss_calculate(scale, translation, predicted_depth, depth_gt, percentile_threshold=90):
    # Apply scale and translation to predicted depth
    scaled_and_translated_predicted = predicted_depth * scale + translation
    
    # Create a mask where GT depth values are not zero
    mask_non_zero = depth_gt != 0
    
    # Apply the mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[mask_non_zero]
    valid_predicted_depth = scaled_and_translated_predicted[mask_non_zero]
    
    # Calculate the L2 loss where GT depth is not zero
    loss_map = (valid_gt_depth - valid_predicted_depth) ** 2
    
    # Calculate the threshold for outliers based on the specified percentile of the loss map
    threshold = np.percentile(loss_map, percentile_threshold)
    
    # Identify outliers within the non-zero mask
    outliers_in_non_zero = loss_map > threshold
    
    # Create the full outlier mask, initializing with False for every element
    outlier_mask = np.zeros_like(depth_gt, dtype=bool)
    outlier_mask[mask_non_zero] = outliers_in_non_zero
    
    # Calculate the number and percentage of values masked out as outliers
    num_masked_outliers = np.sum(outliers_in_non_zero)
    total_values = np.sum(mask_non_zero)
    percent_masked_outliers = (num_masked_outliers / total_values) * 100 if total_values != 0 else 0
    
    # Calculate the mean loss excluding outliers
    valid_loss_map = loss_map[~outliers_in_non_zero]
    
    # Return the mean of squared differences and the outlier mask
    return np.mean(loss_map), outlier_mask

def l2_loss(params, predicted_depth, depth_gt):
    scale, translation = params
    scaled_and_translated_predicted = predicted_depth * scale + translation
    
    # Create a mask where GT depth values are not zero
    mask = depth_gt != 0
    
    # Apply the mask to both GT and predicted depth data
    valid_gt_depth = depth_gt[mask]
    valid_predicted_depth = scaled_and_translated_predicted[mask]
    
    # Calculate the mean of squared differences where GT depth is not zero
    loss_map = (valid_gt_depth - valid_predicted_depth) ** 2
    
    # Return the mean of squared differences and an optional outlier mask
    return np.mean(loss_map), 0  # Outlier mask not used in this simple version


def lstsquare(X, y):
    # Create a mask to filter out positions where y is zero
    mask = y != 0
    
    # Apply the mask to both X and y
    X_masked = X[mask]
    y_masked = y[mask]
    
    # Flatten the masked arrays
    X_flat = X_masked.flatten()
    y_flat = y_masked.flatten()
    
    # Add a column of ones to X_flat for the intercept term
    X_flat_b = np.vstack([X_flat, np.ones(X_flat.shape)]).T
    
    # Solve the least squares problem
    params, residuals, rank, s = np.linalg.lstsq(X_flat_b, y_flat, rcond=None)
    
    # Extract optimal parameters
    optimal_A = params[0]
    optimal_b = params[1]
    
    return optimal_A, optimal_b