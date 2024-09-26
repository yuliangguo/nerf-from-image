import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import time
import random
import cv2
from scipy.spatial.transform import Rotation as R
from datetime import date, datetime
import sys
import math

from torch.utils import tensorboard
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

import arguments
from data import loaders
from data.datasets import WaymoDataset
from lib import pose_utils
from lib import nerf_utils
from lib import utils
from lib import fid
from lib import ops
from lib import metrics
from lib import pose_estimation
from models import generator
from models import discriminator
from models import encoder

"""
    This one is the full evaluation on Waymo given accurate intrinsics provided
"""
# manually record p3d training distribution
p3d_scene_range = 1.4  # pretrained model is based on this scale
p3d_focal_guesses = np.asarray([0.71839845,  1.07731938,  1.32769489,  1.59814608,  1.88348041,  2.27928376,
                            2.82873106,  3.73867059,  5.14416647,  9.12456608, 27.79907417])


def render(target_model,
           height,
           width,
           tform_cam2world,
           focal_length,
           center,
           bbox,
           model_input,
           depth_samples_per_ray,
           randomize=True,
           compute_normals=False,
           compute_semantics=False,
           compute_coords=False,
           extra_model_outputs=[],
           extra_model_inputs={},
           force_no_cam_grad=False):

    # Attention: the focal_length and center need to be assigned propperly for rendering.
    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        # Attention: scene_range seems a sensitive factor, the AABB box limits, in physical scale.
        # The near/far thresh are the intersetion of ins and outs per input ray
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(),
            dataset_config['scene_range'])

    # These depth_values are distance to camera center, not z-buffer. Random sampling within near and far range
    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins,
        ray_directions,
        near_thresh,
        far_thresh,
        depth_samples_per_ray,
        randomize=randomize,
    )

    if force_no_cam_grad:
        query_points = query_points.detach()
        depth_values = depth_values.detach()
        ray_directions = ray_directions.detach()

    if args.use_viewdir:
        viewdirs = ray_directions.unsqueeze(-2)
    else:
        viewdirs = None

    model_outputs = target_model(viewdirs, model_input,
                                 ['sampler'] + extra_model_outputs,
                                 extra_model_inputs)
    radiance_field_sampler = model_outputs['sampler']
    del model_outputs['sampler']

    request_sampler_outputs = ['sigma', 'rgb']
    if compute_normals:
        assert args.use_sdf
        request_sampler_outputs.append('normals')
    if compute_semantics:
        assert args.attention_values > 0
        request_sampler_outputs.append('semantics')
    if compute_coords:
        request_sampler_outputs.append('coords')
    sampler_outputs_coarse = radiance_field_sampler(query_points,
                                                    request_sampler_outputs)
    sigma = sampler_outputs_coarse['sigma'].view(*query_points.shape[:-1], -1)
    rgb = sampler_outputs_coarse['rgb'].view(*query_points.shape[:-1], -1)

    if compute_normals:
        normals = sampler_outputs_coarse['normals'].view(
            *query_points.shape[:-1], -1)
    else:
        normals = None

    if compute_semantics:
        semantics = sampler_outputs_coarse['semantics'].view(
            *query_points.shape[:-1], -1)
    else:
        semantics = None

    if compute_coords:
        coords = sampler_outputs_coarse['coords'].view(*query_points.shape[:-1],
                                                       -1)
    else:
        coords = None

    if args.fine_sampling:
        z_vals = depth_values
        with torch.no_grad():
            weights = nerf_utils.render_volume_density_weights_only(
                sigma.squeeze(-1), ray_origins, ray_directions,
                depth_values).flatten(0, 2)

            # Smooth weights as in EG3D
            weights = F.max_pool1d(weights.unsqueeze(1).float(),
                                   2,
                                   1,
                                   padding=1)
            weights = F.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = nerf_utils.sample_pdf(
                z_vals_mid.flatten(0, 2),
                weights[..., 1:-1],
                depth_samples_per_ray,
                deterministic=not randomize,
            )
            z_samples = z_samples.view(*z_vals.shape[:3], z_samples.shape[-1])

        z_values_sorted, z_indices_sorted = torch.sort(torch.cat(
            (z_vals, z_samples), dim=-1),
                                                       dim=-1)
        query_points_fine = ray_origins[
            ...,
            None, :] + ray_directions[..., None, :] * z_samples[..., :, None]

        sampler_outputs_fine = radiance_field_sampler(query_points_fine,
                                                      request_sampler_outputs)
        sigma_fine = sampler_outputs_fine['sigma'].view(
            *query_points_fine.shape[:-1], -1)
        rgb_fine = sampler_outputs_fine['rgb'].view(
            *query_points_fine.shape[:-1], -1)
        if compute_normals:
            normals_fine = sampler_outputs_fine['normals'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            normals_fine = None
        if compute_semantics:
            semantics_fine = sampler_outputs_fine['semantics'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            semantics_fine = None
        if compute_coords:
            coords_fine = sampler_outputs_fine['coords'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            coords_fine = None

        sigma = torch.cat((sigma, sigma_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  sigma.shape[-1]))
        rgb = torch.cat((rgb, rgb_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  rgb.shape[-1]))
        if normals_fine is not None:
            normals = torch.cat((normals, normals_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      normals.shape[-1]))
        if semantics_fine is not None:
            semantics = torch.cat((semantics, semantics_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      semantics.shape[-1]))
        if coords_fine is not None:
            coords = torch.cat((coords, coords_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      coords.shape[-1]))
        depth_values = z_values_sorted

    if coords is not None:
        semantics = coords

    rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted = nerf_utils.render_volume_density(
        sigma.squeeze(-1),
        rgb,
        ray_origins,
        ray_directions,
        depth_values,
        normals,
        semantics,
        white_background=dataset_config['white_background'])

    # Modified: convert depth_predicted to z-buffer as common depth map setup
    tform_world2cam = pose_utils.invert_space(tform_cam2world)
    view_directions = torch.sum(ray_directions[..., None, :] *
                                tform_world2cam[:, None, None, :3, :3],
                                dim=-1)
    view_points3D = view_directions * depth_predicted.unsqueeze(-1)
    # Revert sign of default flip camera
    depth_predicted = view_points3D[..., -1] * (-1)

    return rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted, model_outputs


def create_model(args, device):
    return generator.Generator(
        args.latent_dim,
        dataset_config['scene_range'],
        attention_values=args.attention_values,
        use_viewdir=use_viewdir,
        use_encoder=args.use_encoder,
        disable_stylegan_noise=args.disable_stylegan_noise,
        use_sdf=args.use_sdf,
        num_classes=None).to(device)


class ParallelModel(nn.Module):

    def __init__(self, resolution, model=None, model_ema=None, lpips_net=None):
        super().__init__()
        self.resolution = resolution
        self.model = model
        self.model_ema = model_ema
        self.lpips_net = lpips_net

    def forward(self,
                tform_cam2world,
                focal,
                center,
                bbox,
                c,
                use_ema=False,
                ray_multiplier=1,
                res_multiplier=1,
                pretrain_sdf=False,
                compute_normals=False,
                compute_semantics=False,
                compute_coords=False,
                encoder_output=False,
                closure=None,
                closure_params=None,
                extra_model_outputs=[],
                extra_model_inputs={},
                force_no_cam_grad=False):
        model_to_use = self.model_ema if use_ema else self.model
        if pretrain_sdf:
            return model_to_use(
                None,
                c,
                request_model_outputs=['sdf_distance_loss', 'sdf_eikonal_loss'])
        if encoder_output:
            return model_to_use.emb(c)

        output = render(model_to_use,
                        int(self.resolution * res_multiplier),
                        int(self.resolution * res_multiplier),
                        tform_cam2world,
                        focal,
                        center,
                        bbox,
                        c,
                        depth_samples_per_ray * ray_multiplier,
                        compute_normals=compute_normals,
                        compute_semantics=compute_semantics,
                        compute_coords=compute_coords,
                        extra_model_outputs=extra_model_outputs,
                        extra_model_inputs=extra_model_inputs,
                        force_no_cam_grad=force_no_cam_grad)
        if closure is not None:
            return closure(
                self, output[0], output[2], output[4], output[-1],
                **closure_params)  # RGB, alpha, semantics, extra_outptus
        else:
            return output


def estimate_poses_batch(target_coords, target_mask, focal_guesses):
    target_mask = target_mask > 0.9
    if focal_guesses is None:
        # Use a large focal length to approximate ortho projection
        is_ortho = True
        focal_guesses = [100.]
    else:
        is_ortho = False

    world2cam_mat, estimated_focal, errors = pose_estimation.compute_pose_pnp(
        target_coords.cpu().numpy(),
        target_mask.cpu().numpy(), focal_guesses)

    if is_ortho:
        # Convert back to ortho
        s = 2 * focal_guesses[0] / -world2cam_mat[:, 2, 3]
        t2 = world2cam_mat[:, :2, 3] * s[..., None]
        world2cam_mat_ortho = world2cam_mat.copy()
        world2cam_mat_ortho[:, :2, 3] = t2
        world2cam_mat_ortho[:, 2, 3] = -10.
        world2cam_mat = world2cam_mat_ortho

    estimated_cam2world_mat = pose_utils.invert_space(
        torch.from_numpy(world2cam_mat).float()).to(target_coords.device)
    estimated_focal = torch.from_numpy(estimated_focal).float().to(
        target_coords.device)
    if is_ortho:
        estimated_cam2world_mat /= torch.from_numpy(
            s[:, None, None]).float().to(estimated_cam2world_mat.device)
        estimated_focal = None

    return estimated_cam2world_mat, estimated_focal, errors


def estimate_poses_batch_new(target_coords, target_mask, intrinsics):
    target_mask = target_mask > 0.9
    # TODO: Skipped those with empty mask predicted from coord_regressor (BootInv)
    if target_mask[0].sum() == 0:
        print('Empty mask detected, skipping...')
        return None, None 

    world2cam_mat, errors = pose_estimation.compute_pose_pnp_new(
        target_coords.cpu().numpy(),
        target_mask.cpu().numpy(), intrinsics)

    estimated_cam2world_mat = pose_utils.invert_space(
        torch.from_numpy(world2cam_mat).float()).to(target_coords.device)

    return estimated_cam2world_mat, errors


def augment_impl(img, pose, focal, p, disable_scale=False, cached_tform=None):
    bs = img.shape[0] if img is not None else pose.shape[0]
    device = img.device if img is not None else pose.device

    if cached_tform is None:
        rot = (torch.rand((bs,), device=device) - 0.5) * 2 * np.pi
        rot = rot * (torch.rand((bs,), device=device) < p).float()

        if disable_scale:
            scale = torch.ones((bs,), device=device)
        else:
            scale = torch.exp2(torch.randn((bs,), device=device) * 0.2)
            scale = torch.lerp(torch.ones_like(scale), scale, (torch.rand(
                (bs,), device=device) < p).float())

        translation = torch.randn((bs, 2), device=device) * 0.1
        translation = torch.lerp(torch.zeros_like(translation), translation,
                                 (torch.rand(
                                     (bs, 1), device=device) < p).float())

        cached_tform = rot, scale, translation
    else:
        rot, scale, translation = cached_tform

    mat = torch.zeros((bs, 2, 3), device=device)
    mat[:, 0, 0] = torch.cos(rot)
    mat[:, 0, 1] = -torch.sin(rot)
    mat[:, 0, 2] = translation[:, 0]
    mat[:, 1, 0] = torch.sin(rot)
    mat[:, 1, 1] = torch.cos(rot)
    mat[:, 1, 2] = -translation[:, 1]
    if img is not None:
        mat_scaled = mat.clone()
        mat_scaled *= scale[:, None, None]
        mat_scaled[:, :, 2] = torch.sum(mat[:, :2, :2] *
                                        mat_scaled[:, :, 2].unsqueeze(-2),
                                        dim=-1)
        grid = F.affine_grid(mat_scaled, img.shape, align_corners=False)
        if dataset_config['white_background']:
            assert not args.supervise_alpha
            img = img - 1  # Adjustment for white background
        img_transformed = F.grid_sample(img,
                                        grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)
        if dataset_config['white_background']:
            img_transformed = img_transformed + 1  # Adjustment for white background
    else:
        img_transformed = None

    if pose is not None:
        M = torch.eye(4, device=device).unsqueeze(0).expand(mat.shape[0], 4,
                                                            4).contiguous()
        M[:, :2, :2] = mat[:, :2, :2]
        if focal is not None:
            focal = focal / scale
        pose = pose @ M.transpose(-2, -1)
        if focal is None:
            pose[:, :3, :3] *= scale[:, None, None]
            pose[:, 3:4, 3:4] *= scale[:, None, None]

        # Apply translation
        pose_orig = pose
        cam_inverted = pose_utils.invert_space(pose)
        if focal is not None:
            cam_inverted[:, :2, 3] -= translation * (-cam_inverted[:, 2:3, 3] /
                                                     (2 * focal[:, None]))
        else:
            cam_inverted[:, :2, 3] -= translation * pose_orig[:, 3:4, 3]
        pose = pose_utils.invert_space(cam_inverted)
        if focal is None:
            pose[:, :3, :3] *= pose_orig[:, 3:4, 3:4]
            pose[:, 3, 3] *= pose_orig[:, 3, 3]

    return img_transformed, pose, focal, cached_tform


def augment(img,
            pose,
            focal,
            p,
            disable_scale=False,
            cached_tform=None,
            return_tform=False):
    if p == 0 and cached_tform is None:
        return img, pose, focal

    assert img is None or pose is None or img.shape[0] == pose.shape[0]

    # Standard augmentation
    img_new, pose_new, focal_new, tform = augment_impl(img, pose, focal, p,
                                                       disable_scale,
                                                       cached_tform)

    if return_tform:
        return img_new, pose_new, focal_new, tform
    else:
        return img_new, pose_new, focal_new


def optimize_iter(module, rgb_predicted, acc_predicted,
                  semantics_predicted, extra_model_outputs, target_img,
                  cam, focal):
    target = target_img[..., :3]

    rgb_predicted_for_loss = rgb_predicted
    target_for_loss = target
    loss = 0.
    if loss_to_use in ['vgg_nocrop', 'vgg', 'mixed']:
        rgb_predicted_for_loss_aug = rgb_predicted_for_loss.permute(
            0, 3, 1, 2)
        target_for_loss_aug = target_for_loss.permute(0, 3, 1, 2)
        num_augmentations = 0 if loss_to_use == 'vgg_nocrop' else 15
        if num_augmentations > 0:
            predicted_target_cat = torch.cat(
                (rgb_predicted_for_loss_aug, target_for_loss_aug),
                dim=1)
            predicted_target_cat = predicted_target_cat.unsqueeze(
                1).expand(-1, num_augmentations, -1, -1,
                          -1).contiguous().flatten(0, 1)
            predicted_target_cat, _, _ = augment(
                predicted_target_cat, None, None, 1.0)
            rgb_predicted_for_loss_aug = torch.cat(
                (rgb_predicted_for_loss_aug,
                 predicted_target_cat[:, :3]),
                dim=0)
            target_for_loss_aug = torch.cat(
                (target_for_loss_aug, predicted_target_cat[:, 3:]),
                dim=0)
        loss = loss + module.lpips_net(
            rgb_predicted_for_loss_aug, target_for_loss_aug
        ).mean() * rgb_predicted.shape[
                   0]  # Disjoint samples, sum instead of average over batch
    if loss_to_use in ['l1', 'mixed']:
        loss = loss + F.l1_loss(rgb_predicted_for_loss, target_for_loss
                                ) * rgb_predicted.shape[0]
    if loss_to_use == 'mse':
        loss = F.mse_loss(rgb_predicted_for_loss,
                          target_for_loss) * rgb_predicted.shape[0]

    if loss_to_use == 'mixed':
        loss = loss / 2  # Average L1 and VGG

    with torch.no_grad():
        # TODO: also add mask here?
        psnr_monitor = metrics.psnr(rgb_predicted[..., :3] / 2 + 0.5,
                                    target[..., :3] / 2 + 0.5)
        lpips_monitor = module.lpips_net(
            rgb_predicted[..., :3].permute(0, 3, 1, 2),
            target[..., :3].permute(0, 3, 1, 2),
            normalize=False)

    return loss, psnr_monitor, lpips_monitor, rgb_predicted


def evaluate_inversion(obj_idx, it, out_dir, target_img_fid_, target_center_fid, target_bbox_fid, export_sample=False, inception_net=None):
    item = report[it]
    # TODO: Now just assume z_, z0_, R_, s_, t2_ in right number before calling this function
    item['ws'].append(z_.detach().cpu() * lr_gain_z)
    if z0_ is not None:
        item['z0'].append(z0_.detach().cpu())
    item['R'].append(R_.detach().cpu())
    item['s'].append(s_.detach().cpu())
    item['t2'].append(t2_.detach().cpu())

    # Compute metrics for report
    cam, focal = pose_utils.pose_to_matrix(
        z0_.detach() if z0_ is not None else None,
        t2_.detach(),
        s_.detach(),
        F.normalize(R_.detach(), dim=-1),
        # camera_flipped=False)
        camera_flipped=dataset_config['camera_flipped'])
    rgb_predicted, depth_predicted, acc_predicted, normals_predicted, semantics_predicted, extra_model_outputs = model_to_call(
        cam,
        focal,
        target_center_fid,
        target_bbox_fid,
        z_.detach() * lr_gain_z,
        use_ema=True,
        # compute_normals=args.use_sdf and idx == 0,
        compute_normals=args.use_sdf,
        compute_semantics=args.attention_values > 0,
        force_no_cam_grad=True,
        extra_model_outputs=['attention_values']
        if args.attention_values > 0 else [],
        extra_model_inputs={
            k: v.detach() for k, v in extra_model_inputs.items()
        },
    )

    rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                        2).clamp_(
        -1, 1)
    target_perm = target_img_fid_.permute(0, 3, 1, 2)

    if export_sample:
        with torch.no_grad():
            demo_img = target_perm[:, :3]
            if use_pose_regressor and target_mask is not None:
                coords_img = (
                        target_coords.permute(0, 3, 1, 2)
                        * target_mask.unsqueeze(1))
                coords_img /= dataset_config['scene_range']
                coords_img.clamp_(-1, 1)
                coords_img_empty = torch.full_like(coords_img, 0)
                if dataset_config['white_background']:
                    coords_img += 1 - target_mask.unsqueeze(1)
                    coords_img_empty += 1
                if args.init_pose_type == "external":
                    demo_img = torch.cat((demo_img, coords_img_empty), dim=3)
                else:
                    demo_img = torch.cat((demo_img, coords_img), dim=3)
            demo_img = torch.cat((demo_img, rgb_predicted_perm), dim=3)

            # prepare depth map visualization -- normalize with fg values, range to [-1, 1]
            depth_fg = depth_predicted[acc_predicted > 0.5]
            # using fixed range might be better for inaccurate mask
            depth_vis = (depth_predicted - torch.median(depth_fg)) / 5
            if dataset_config['white_background']:
                depth_vis[acc_predicted < 0.95] = 1.0  # white bg
            else:
                depth_vis[acc_predicted < 0.95] = 0.0  # grey bg
            depth_vis = depth_vis.unsqueeze(-1).repeat(1, 1, 1, 3)
            demo_img = torch.cat((demo_img, depth_vis.permute(0, 3, 1, 2)), dim=3)

            if normals_predicted is not None:
                demo_img = torch.cat(
                    (demo_img, normals_predicted.permute(0, 3, 1, 2)),
                    dim=3)

    # psnr_mask = torch.logical_and(target_mask_input.unsqueeze(1) > 0.5, acc_predicted > 0.5)
    # if psnr_mask.sum() == 0:
    psnr_mask = target_mask_input.unsqueeze(1)
    psnr = metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
                        target_perm[:, :3] / 2 + 0.5,
                        reduction='none',  # ).cpu()
                        mask=psnr_mask.repeat(1, 3, 1, 1)).cpu()

    item['psnr'].append(psnr)
    item['ssim'].append(
        metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
                     target_perm[:, :3] / 2 + 0.5,
                     reduction='none').cpu())
    # if dataset_config['has_mask']:
    #     item['iou'].append(
    #         metrics.iou(acc_predicted,
    #                     target_perm[:, 3],
    #                     reduction='none').cpu())
    item['lpips'].append(
        loss_fn_lpips(rgb_predicted_perm[:, :3],
                      target_perm[:, :3],
                      normalize=False).flatten().cpu())
    if not args.inv_export_demo_sample:
        item['inception_activations_front'].append(
            torch.FloatTensor(
                fid.forward_inception_batch(
                    inception_net,
                    rgb_predicted_perm[:, :3] / 2 + 0.5)))
    # if not (args.dataset == 'p3d_car' and use_testset):
    # Ground-truth poses are not available on P3D Car (test set)
    depth_error = torch.mean(torch.abs(gt_depth - depth_predicted)[torch.logical_and(gt_depth_mask, target_mask_input)])
    item['depth_error'].append(depth_error)

    rot_error = pose_utils.rotation_matrix_distance(cam, gt_cam2world_mat)
    item['rot_error'].append(rot_error)

    # Trans_error need to be converted back to object space to compute, or the rotation entanglement makes it larger
    trans_error = torch.sqrt(torch.sum((pose_utils.invert_space(cam)[:, :3, 3] -
                                        pose_utils.invert_space(gt_cam2world_mat)[:, :3, 3])**2))
    item['trans_error'].append(trans_error)

    # just perturb the original view
    angle_lim = np.pi * 0.2
    rotvec_rand = [random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim),
                   random.uniform(-angle_lim, angle_lim)]
    R_rand = R.from_rotvec(rotvec_rand).as_matrix()
    target_tform_cam2world = cam.clone()
    target_tform_world2cam_perm = pose_utils.invert_space(target_tform_cam2world)
    target_tform_world2cam_perm[0, :3, :3] = target_tform_world2cam_perm[0, :3, :3] @ torch.FloatTensor(R_rand).to(device)
    target_tform_cam2world_perm = pose_utils.invert_space(target_tform_world2cam_perm)
    # target_focal_perm = None
    # target_focal_perm = focal * random.uniform(0.7, 2)
    target_focal_perm = focal
    target_center_perm = target_center
    target_bbox_perm = None

    rgb_predicted, depth_predicted, acc_predicted, normals_predicted, semantics_predicted, _ = model_to_call(
        target_tform_cam2world_perm,
        target_focal_perm,
        target_center_perm,
        target_bbox_perm,
        z_.detach() * lr_gain_z,
        use_ema=True,
        compute_normals=args.use_sdf,
        compute_semantics=args.attention_values > 0 and idx == 0,
        force_no_cam_grad=True,
        extra_model_inputs={
            k: v.detach() for k, v in extra_model_inputs.items()
        },
    )
    rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                        2).clamp(-1, 1)

    print(f'it{it}: psnr: {psnr.item()}, depth error: {depth_error.item()}, '
          f'rot error: {rot_error.item()}, trans error: {trans_error.item()}')

    if export_sample:
        with torch.no_grad():
            demo_img = torch.cat((demo_img, rgb_predicted_perm), dim=3)

            # prepare depth map visualization -- normalize with fg values, range to [-1, 1]
            depth_fg = depth_predicted[acc_predicted > 0.5]
            # using fixed range might be better for inaccurate mask
            depth_vis = (depth_predicted - torch.median(depth_fg)) / 5
            if dataset_config['white_background']:
                depth_vis[acc_predicted < 0.95] = 1.0  # white bg
            else:
                depth_vis[acc_predicted < 0.95] = 0.0  # grey bg
            depth_vis = depth_vis.unsqueeze(-1).repeat(1, 1, 1, 3)
            demo_img = torch.cat((demo_img, depth_vis.permute(0, 3, 1, 2)), dim=3)

            if normals_predicted is not None:
                demo_img = torch.cat(
                    (demo_img, normals_predicted.permute(0, 3, 1, 2)),
                    dim=3)

            # print the eval message on the demo image
            eval_str = 'PSNR: {:.2f},  Depth err: {:.2f}, R err: {:.2f}, T err: {:.2f}'.format(
                psnr.item(), depth_error.item(), rot_error.item(), trans_error.item())
            demo_img = ((demo_img.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() / 2 + 0.5) * 255).astype(np.uint8)
            demo_img = cv2.putText(demo_img.copy(), eval_str, (260, 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0))
            demo_img = torch.from_numpy(demo_img.astype(np.float32) / 255).unsqueeze(0).permute(0, 3, 1, 2)

            # Move the saving code before
            utils.mkdir(out_dir)
            out_fname = f'demo_obj{obj_idx}_{it}it.png'
            out_path = os.path.join(out_dir, out_fname)
            print('Saving demo output to', out_path)
            torchvision.utils.save_image(demo_img,
                                         out_path,
                                         nrow=1,
                                         padding=0)


if __name__ == '__main__':
    # scene_range should match the target testing dataset. NeRF model will scale based on it
    dataset_config = {'scene_range': 3.0, 'camera_flipped': True, 'white_background': True}
    args = arguments.parse_args()
    # args.resume_from = 'g_imagenet_car_pretrained'
    # args.inv_loss = 'vgg'  # vgg / l1 / mse
    # args.fine_sampling = True
    # no_optimize_pose = args.inv_no_optimize_pose
    # no_optimize_pose = False  # for debugging: tmp debug only the nerf given perfect pose
    # init_pose_type = 'external'  # pnp / gt / external
    gpu_ids = [0]
    # max_num_samples = 1225
    utils.fix_random_seed(543)

    exp_name = f'waymo_init_{args.init_pose_type}_opt_pose_{args.no_optimize_pose==False}' + datetime.now().strftime('_%Y_%m_%d_%H')
    # exp_name = f'waymo_init_{init_pose_type}_opt_pose_{no_optimize_pose==False}' + date.today().strftime('_%Y_%m_%d')
    out_dir = os.path.join('outputs', exp_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print(f'Saving results to: {out_dir}')
    out_log = os.path.join(out_dir, 'log.txt')

    # waymo_data_dir = '/media/yuliangguo/data_ssd_4tb/Datasets/Waymo_validation_set_DEVIANT'

    # upnerf result file
    # external_pose_file = '../nerf-auto-driving/exps_nuscenes_supnerf/vehicle.car.v1.0-trainval.use_instance.bsize24.e_rate1.0_2023_02_15_new_infer/test_waymo_opt_pose_1_poss_err_full_reg_iters_3_epoch_39_wt_dep/codes+poses.pth'
    # external_pose_file = None

    waymo_dataset = WaymoDataset(
        args.waymo_data_dir,
        debug=False,
        external_pose_file=args.external_pose_file,
        white_bkgd=dataset_config['white_background']
    )

    waymo_loader = DataLoader(waymo_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    """
        got the minimal viable portion to model to run
    """
    args.gpus = 1 if args.gpus >= 1 else 0
    args.inv_export_demo_sample = True
    if args.inv_export_demo_sample:
        args.run_inversion = True
    # gpu_ids = list(range(args.gpus))

    if args.gpus > 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}') 
    else:
        device = torch.device('cpu')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    experiment_name = arguments.suggest_experiment_name(args)
    resume_from = None
    log_dir = 'gan_logs'
    report_dir = 'reports'
    file_dir = 'gan_checkpoints'

    checkpoint_dir = os.path.join(args.root_path, file_dir, experiment_name)
    report_dir = os.path.join(args.root_path, report_dir)
    print('Saving inversion reports to', report_dir)
    utils.mkdir(report_dir)

    writer = None  # Instantiate later

    # Load latest
    print('Attempting to load latest checkpoint...')
    last_checkpoint_dir = os.path.join(args.root_path, 'gan_checkpoints',
                                       args.resume_from,
                                       'checkpoint_latest.pth')

    if utils.file_exists(last_checkpoint_dir):
        print('Resuming from manual checkpoint', last_checkpoint_dir)
        with utils.open_file(last_checkpoint_dir, 'rb') as f:
            resume_from = torch.load(f, map_location='cpu')
    else:
        raise ValueError(
            f'Specified checkpoint {args.resume_from} does not exist!')

    if resume_from is not None:
        print('Checkpoint iteration:', resume_from['iteration'])
        if 'fid_untrunc' in resume_from:
            print('Checkpoint unconditional FID:', resume_from['fid_untrunc'])
            print('Checkpoint unconditional FID (best):', resume_from['best_fid'])

    if args.attention_values > 0:
        color_palette = utils.get_color_palette(args.attention_values).to(device)
    else:
        color_palette = None

    evaluation_res = args.resolution
    # inception_net = fid.init_inception('tensorflow').to(device).eval()

    random_seed = 1234
    n_images_fid_max = 8000  # Matches Pix2NeRF evaluation protocol

    random_generator = torch.Generator()
    random_generator.manual_seed(random_seed)

    batch_size = args.batch_size
    use_viewdir = args.use_viewdir
    supervise_alpha = args.supervise_alpha
    use_encoder = args.use_encoder  # if the generator run encoder on image
    use_r1 = args.r1 > 0
    use_tv = args.tv > 0
    use_entropy = args.entropy > 0

    # Number of depth samples along each ray
    depth_samples_per_ray = 64
    if not args.fine_sampling:
        depth_samples_per_ray *= 2  # More fair comparison

    if args.use_encoder or args.run_inversion:
        loss_fn_lpips = metrics.LPIPSLoss().to(device)
    else:
        loss_fn_lpips = None

    """
        prepare generator model
    """
    if args.run_inversion:
        model = None
    else:
        model = create_model(args, device)

    model_ema = create_model(args, device)
    model_ema.eval()
    model_ema.requires_grad_(False)
    if model is not None:
        model_ema.load_state_dict(model.state_dict())

    parallel_model = nn.DataParallel(
        ParallelModel(args.resolution,
                      model=model,
                      model_ema=model_ema,
                      lpips_net=loss_fn_lpips), gpu_ids).to(device)

    total_params = 0
    for param in model_ema.parameters():
        total_params += param.numel()
    print('Params G:', total_params / 1000000, 'M')

    # Seed CUDA RNGs separately
    seed_generator = np.random.RandomState(random_seed)
    for device_id in gpu_ids:
        with torch.cuda.device(device_id):
            gpu_seed = int.from_bytes(np.random.bytes(4), 'little', signed=False)
            torch.cuda.manual_seed(gpu_seed)

    if resume_from is not None:
        print('Loading specified checkpoint...')
        if model is not None and 'model' in resume_from:
            model.load_state_dict(resume_from['model'])
        model_ema.load_state_dict(resume_from['model_ema'])

        if 'iteration' in resume_from:
            i = resume_from['iteration']
            print('Resuming GAN from iteration', i)
        else:
            i = args.iterations

    # Global config
    use_testset = args.inv_use_testset
    use_pose_regressor = True
    use_latent_regressor = True
    loss_to_use = args.inv_loss
    lr_gain_z = args.inv_gain_z
    inv_no_split = args.inv_no_split
    batch_size = 1

    """
        load pretrained coord_regressor
    """
    with utils.open_file('coords_checkpoints/g_p3d_car_pretrained/c_it300000_latest.pth',
                         'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    regress_pose = True
    regress_latent = True
    regress_separate = args.inv_use_separate

    coord_regressor = encoder.BootstrapEncoder(
        args.latent_dim,
        pose_regressor=regress_pose,
        latent_regressor=regress_latent,
        separate_backbones=regress_separate,
        pretrained_model_path=os.path.join(args.root_path,
                                           'coords_checkpoints'),
        pretrained=checkpoint is None,
    ).to(device)

    coord_regressor = nn.DataParallel(coord_regressor, gpu_ids)
    coord_regressor.requires_grad_(True)

    # Resume if cache is available
    if checkpoint is not None:
        coord_regressor.load_state_dict(checkpoint['model_coord'])


    """
        other initial parameters
    """

    checkpoint_steps = [0, 20, 50]

    report = {
        step: {
            'ws': [],
            'z0': [],
            'R': [],
            's': [],
            't2': [],
            'psnr': [],
            'psnr_random': [],
            'lpips': [],
            'lpips_random': [],
            'ssim': [],
            'ssim_random': [],
            'iou': [],
            'depth_error': [],
            'rot_error': [],
            'trans_error': [],
            'inception_activations_front': [],  # Front view
            'inception_activations_random': [],  # Random view
        } for step in checkpoint_steps
    }

    with torch.no_grad():
        z_avg = model_ema.mapping_network.get_average_w()

    report_checkpoint_path = os.path.join(out_dir, 'report_checkpoint.pth')
    print('Running...')
    # deal with each detected object in the image
    for idx, batch_data in enumerate(waymo_loader):
        # only evaluate a subset to save time
        if idx >= args.max_num_samples and args.max_num_samples > 0:
            break
        t1 = time.time()

        target_img = batch_data['img_batch'].to(device)
        target_mask_input = batch_data['mask_batch'].to(device)
        target_img_fid_ = target_img  # Target for evaluation (front view -- always cropped)
        target_focal = batch_data['K_batch'][0, 0, 0].to(device)
        target_center = batch_data['K_batch'][:, :2, 2].to(device) + 0.5  # this is for the rendering range in the given image, None for full patch
        target_bbox = None  # this is for the rendering range in the given image, in pixels. None for full patch

        target_center_fid = batch_data['K_batch'][:, :2, 2].to(device) + 0.5
        target_bbox_fid = None

        wlh_batch = batch_data['wlh']
        gt_world2cam_mat = torch.eye(4).unsqueeze(0)
        gt_world2cam_mat[0, :3, :] = utils.obj_pose_kitti2nusc(batch_data['obj_poses'], wlh_batch[:, 2])

        gt_cam2world_mat = pose_utils.invert_space(gt_world2cam_mat)
        nusc2shapenet = torch.FloatTensor([[0, 1, 0, 0],
                                           [-1, 0, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        gt_cam2world_mat[0] = nusc2shapenet @ gt_cam2world_mat[0]
        gt_cam2world_mat = gt_cam2world_mat.to(device)
        if dataset_config['camera_flipped']:
            # gt_cam2world_mat[:, :3, 1:] *= -1
            gt_cam2world_mat[:, :3, 1:3] *= -1
        gt_depth = batch_data['depth_batch'].to(device)
        gt_depth_mask = gt_depth > 0

        z_ = z_avg.clone().expand(1, -1, -1).contiguous()

        # Modified: to use the actual known intrinsics
        with torch.no_grad():
            coord_regressor_img = target_img[..., :3].permute(0, 3, 1, 2)
            # convert back to grey bg since coord regressor was trained with that
            if dataset_config['white_background']:
                # convert to white background to grey to match the trained model
                coord_regressor_img = coord_regressor_img.clone()
                coord_regressor_img += (target_mask_input.unsqueeze(1) - 1) * 0.5
            target_coords, target_mask, target_w = coord_regressor.module(
                coord_regressor_img)
            # modified: adjust the scale to target dataset
            target_coords *= (dataset_config['scene_range'] / p3d_scene_range)

            # if use_pose_regressor:
            assert target_coords is not None
            estimated_cam2world_mat, _ = estimate_poses_batch_new(
                target_coords, target_mask, batch_data['K_batch'])
            if estimated_cam2world_mat is None:
                continue
            if use_latent_regressor:
                assert target_w is not None
                z_.data[:] = target_w

        # For Debugging: tmp debug only the nerf given perfect pose
        if args.init_pose_type == 'gt':
            target_tform_cam2world = gt_cam2world_mat
        elif args.init_pose_type == 'external':
            ext_world2cam_mat = torch.eye(4).unsqueeze(0)
            ext_world2cam_mat[0, :3, :] = batch_data['obj_poses_ext'][0, 1]
            ext_cam2world_mat = pose_utils.invert_space(ext_world2cam_mat)
            nusc2shapenet = torch.FloatTensor([[0, 1, 0, 0],
                                               [-1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]])
            ext_cam2world_mat[0] = nusc2shapenet @ ext_cam2world_mat[0]
            if dataset_config['camera_flipped']:
                ext_cam2world_mat[:, :3, 1:3] *= -1
            target_tform_cam2world = ext_cam2world_mat.to(device)
        else:
            target_tform_cam2world = estimated_cam2world_mat

        # # For Debugging: printing the estimated pose and
        # print('estimated cam pose')
        # print(target_tform_cam2world[0].cpu().numpy())
        # print('gt cam pose converted')
        # print(gt_cam2world_mat[0].cpu().numpy())
        # print('estimated object pose')
        # print(pose_utils.invert_space(target_tform_cam2world)[0].cpu().numpy())
        # print('gt object pose origin')
        # print(pose_utils.invert_space(gt_cam2world_mat)[0].cpu().numpy())

        if inv_no_split:
            z_ = z_.mean(dim=1, keepdim=True)

        z_ /= lr_gain_z
        z_ = z_.requires_grad_()

        # might be working, passing target_focal as None, so not to optimize
        z0_, t2_, s_, R_ = pose_utils.matrix_to_pose(
            target_tform_cam2world,
            target_focal,
            # camera_flipped=False)
            camera_flipped=dataset_config['camera_flipped'])

        if not args.no_optimize_pose:
            t2_.requires_grad_()
            s_.requires_grad_()
            R_.requires_grad_()
        # if z0_ is not None:
        #     if not no_optimize_pose:
        #         z0_.requires_grad_()
        #     param_list = [z_, z0_, R_, s_, t2_]
        #     param_names = ['z', 'f', 'R', 's', 't']
        # else:
        # modified: never optimize focal length, since given the true
        param_list = [z_, R_, s_, t2_]
        param_names = ['z', 'R', 's', 't']
        if args.no_optimize_pose:
            param_list = param_list[:1]
            param_names = param_names[:1]

        extra_model_inputs = {}
        optimizer = torch.optim.Adam(param_list, lr=2e-3, betas=(0.9, 0.95))
        grad_norms = []
        for _ in range(len(param_list)):
            grad_norms.append([])

        model_to_call = parallel_model if z_.shape[
                                              0] > 1 else parallel_model.module

        psnrs = []
        lpipss = []
        rot_errors = []
        niter = max(checkpoint_steps)

        if 0 in checkpoint_steps:
            evaluate_inversion(idx, 0, out_dir, target_img_fid_, target_center_fid, target_bbox_fid,
                               (args.inv_export_demo_sample
                                and max(checkpoint_steps) == 0))

        for it in range(niter):
            cam, focal = pose_utils.pose_to_matrix(
                z0_,
                t2_,
                s_,
                F.normalize(R_, dim=-1),
                # camera_flipped=False)
                camera_flipped=dataset_config['camera_flipped'])

            # Attention: need to assign focal and target_center properly for given camera K
            loss, psnr_monitor, lpips_monitor, rgb_predicted = model_to_call(
                cam,
                focal,
                target_center,
                target_bbox,
                z_ * lr_gain_z,
                use_ema=True,
                # ray_multiplier=1 if args.fine_sampling else 2,
                # res_multiplier=1,
                compute_normals=False and args.use_sdf,
                force_no_cam_grad=args.no_optimize_pose,
                closure=optimize_iter,
                closure_params={
                    'target_img': target_img,
                    'cam': cam,
                    'focal': focal
                },
                extra_model_inputs=extra_model_inputs,
            )
            normal_map = None
            loss = loss.sum()
            psnr_monitor = psnr_monitor.mean()
            lpips_monitor = lpips_monitor.mean()
            # if writer is not None and idx == 0:
            #     writer.add_scalar('monitor_b0/psnr', psnr_monitor.item(), it)
            #     writer.add_scalar('monitor_b0/lpips', lpips_monitor.item(), it)
            #     rot_error = pose_utils.rotation_matrix_distance(
            #         cam, gt_cam2world_mat).mean().item()
            #     rot_errors.append(rot_error)
            #     writer.add_scalar('monitor_b0/rot_error', rot_error, it)

            if args.use_sdf and normal_map is not None:
                rgb_predicted = torch.cat(
                    (rgb_predicted.detach(), normal_map.detach()), dim=-2)

            loss.backward()
            for i, param in enumerate(param_list):
                if param.grad is not None:
                    grad_norms[i].append(param.grad.norm().item())
                else:
                    grad_norms[i].append(0.)
            optimizer.step()
            optimizer.zero_grad()
            R_.data[:] = F.normalize(R_.data, dim=-1)
            if z0_ is not None:
                z0_.data.clamp_(-4, 4)
            s_.data.abs_()

            # if args.inv_export_demo_sample:
            #     print(it + 1, '/', max(checkpoint_steps))
            if it + 1 in report:
                evaluate_inversion(idx + 1, it + 1, out_dir,
                                   target_img_fid_, target_center_fid, target_bbox_fid,
                                   (args.inv_export_demo_sample and it + 1 == max(checkpoint_steps)))

        t2 = time.time()
        print(
            f'[{idx+1}/{len(waymo_loader)}] Finished batch in {t2 - t1} s ({(t2 - t1)} s/img)'
        )

        if (idx+1) % 20 == 0:
            # Save report checkpoint
            with utils.open_file(report_checkpoint_path, 'wb') as f:
                torch.save({
                    'report': report,
                    'idx': idx,
                }, f)
            with open(out_log, 'w') as file:
                print('====================================================')
                file.write('====================================================\n')
                for it in checkpoint_steps:
                    avg_psnr = torch.mean(torch.stack(report[it]['psnr']))
                    depth_errors = torch.stack(report[it]['depth_error'])
                    avg_depth_error = torch.mean(depth_errors[~torch.isnan(depth_errors)])
                    avg_R_err = torch.mean(torch.stack(report[it]['rot_error']))
                    avg_T_err = torch.mean(torch.stack(report[it]['trans_error']))
                    # avg_ssim = torch.mean(torch.stack(report[it]['ssim']))
                    # avg_lpips = torch.mean(torch.stack(report[it]['lpips']))

                    out_string = f'it{it}: psnr avg: {avg_psnr.item()}, depth error avg: {avg_depth_error.item()}, rot error avg: {avg_R_err.item()}, trans error avg: {avg_T_err.item()}'
                    print(out_string)
                    file.write(out_string + '\n')
                print('====================================================')
                file.write('====================================================\n')

    # final save report
    with utils.open_file(report_checkpoint_path, 'wb') as f:
        torch.save({
            'report': report,
            'idx': len(waymo_loader),
        }, f)
    with open(out_log, 'w') as file:
        print('====================================================')
        file.write('====================================================\n')
        for it in checkpoint_steps:
            avg_psnr = torch.mean(torch.stack(report[it]['psnr']))
            depth_errors = torch.stack(report[it]['depth_error'])
            avg_depth_error = torch.mean(depth_errors[~torch.isnan(depth_errors)])
            avg_R_err = torch.mean(torch.stack(report[it]['rot_error']))
            avg_T_err = torch.mean(torch.stack(report[it]['trans_error']))
            # avg_ssim = torch.mean(torch.stack(report[it]['ssim']))
            # avg_lpips = torch.mean(torch.stack(report[it]['lpips']))

            out_string = f'it{it}: psnr avg: {avg_psnr.item()}, depth error avg: {avg_depth_error.item()}, rot error avg: {avg_R_err.item()}, trans error avg: {avg_T_err.item()}'
            print(out_string)
            file.write(out_string + '\n')
        print('====================================================')
        file.write('====================================================\n')

