import os
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time
import sys
import math
from PIL import Image
import cv2

from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.nuscenes import NuScenes

from torch.utils import tensorboard
from torch import nn
from tqdm import tqdm

import arguments
from data import loaders
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


dataset_config = {'scene_range': 1.4, 'camera_flipped': True, 'white_background': False}


def square_bbox(bbox):
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round((maxdim - bwidth) / 2.0))
    dh_b_2 = int(round((maxdim - bheight) / 2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def crop(img, bbox, bgval=0):
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc)) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg,
    x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src,
                              x_min_src:x_max_src, :]
    return img_out


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) *
                         scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]),
                         interpolation=cv2.INTER_AREA)
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def roi_resize(roi, ratio=1.0):
    min_x, min_y, max_x, max_y = roi
    # enlarge pred_box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    box_w = max_x - min_x
    box_h = max_y - min_y
    min_x = center_x - box_w / 2 * ratio
    max_x = center_x + box_w / 2 * ratio
    min_y = center_y - box_h / 2 * ratio
    max_y = center_y + box_h / 2 * ratio
    roi = [min_x, min_y, max_x, max_y]
    return roi


def get_mask_occ_from_ins(masks, tgt_ins_id):
    """
        Prepare occupancy mask:
            target object: 1
            background: -1 (not likely to occlude the object)
            occluded the instance: 0 (seems only able to reason the occlusion by foreground)
    """
    tgt_mask = masks[tgt_ins_id]
    mask_occ = np.zeros_like(tgt_mask).astype(np.int32)
    mask_union = np.sum(np.asarray(masks), axis=0)

    mask_occ[mask_union == 0] = -1
    mask_occ[tgt_mask > 0] = 1
    return mask_occ


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 nusc_data_dir,
                 nusc_seg_dir,
                 nusc_version,
                 split='train',
                 img_size=128,
                 debug=False
                 ):

        self.seg_cat = 'car'
        self.nusc_data_dir = nusc_data_dir
        self.nusc_seg_dir = nusc_seg_dir
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_data_dir, verbose=True)
        self.img_size = img_size

        print('Preparing camera data dictionary for fast retrival given image name')
        self.cam_data_dict = {}
        for sample in self.nusc.sample_data:
            if 'CAM' in sample['channel']:
                self.cam_data_dict[os.path.basename(sample['filename'])] = sample

        self.debug = debug

    def get_objects_in_image(self, filename):
        """
            Output mask-rcnn masks and boxes per image
            TODO: add the associated GT object pose, lidar measures
        """
        if filename not in self.cam_data_dict.keys():
            print(f'Target image file {filename} does not contain valid annotations')
            return None

        cam_data = self.cam_data_dict[filename]
        # load image, 2D boxes and masks, only need to get K from nusc
        impath, _, camera_intrinsic = self.nusc.get_sample_data(cam_data['token'], box_vis_level=BoxVisibility.ANY)

        # load image
        img_org = Image.open(impath)
        img_org = np.asarray(img_org)

        # load mask-rcnn predicted instance masks and 2D boxes
        cam = cam_data['channel']
        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        rois = []
        for ii in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + f'_{ii}.png')
            mask = np.asarray(Image.open(mask_file))
            if self.seg_cat in preds['labels'][ii]:
                ins_masks.append(mask)
                box_2d = preds['boxes'][ii]
                # enlarge pred_box
                # box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)
                rois.append(box_2d)
        if len(rois) == 0:
            print('No valid objects found in the Image!')
            return None

        masks_occ = []
        for ii in range(0, len(ins_masks)):
            mask_occ = get_mask_occ_from_ins(ins_masks, ii)
            masks_occ.append(mask_occ)

        # Need to predict whl from trained model
        if self.debug:
            self.nusc.render_sample_data(cam_data['token'])
            plt.show()

        # output data
        sample_data = {}
        sample_data['img_org'] = torch.from_numpy(img_org.astype(np.float32) / 255.)
        sample_data['masks_occ'] = torch.from_numpy(np.asarray(masks_occ).astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(rois).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(camera_intrinsic.astype(np.float32))

        # prepared square boxes and crops
        images = []
        masks = []
        bboxes = []
        Ks = []

        for ii, bbox in enumerate(rois):
            bbox = square_bbox(bbox)
            max_res = max(img_org.shape[0], img_org.shape[1])

            K = camera_intrinsic.copy()
            # important! sfm_pose must not be overwritten -- it is already in the correct reference frame
            img = img_org.astype(np.float32).copy()/255.
            img = crop(img, bbox, bgval=1)
            mask = ins_masks[ii].copy()[:, :, None]/255
            mask = crop(mask, bbox, bgval=0)
            K[0, 2] -= bbox[0]
            K[1, 2] -= bbox[1]

            # Scale image so largest bbox size is img_size
            bwidth = np.shape(img)[0]
            bheight = np.shape(img)[1]
            scale = self.img_size / float(max(bwidth, bheight))
            img, _ = resize_img(img, scale)
            mask, _ = resize_img(mask, scale)
            K[:2, :] *= scale

            # Finally transpose the image to 3xHxW
            img = np.transpose(img, (2, 0, 1))

            mask = mask[None, :, :]
            img = img * 2 - 1
            img *= mask
            img = torch.FloatTensor(img).permute(1, 2, 0)

            images.append(img.unsqueeze(0))
            masks.append(torch.FloatTensor(mask))
            bboxes.append(torch.FloatTensor(bbox).unsqueeze(0))
            Ks.append(torch.FloatTensor(K).unsqueeze(0))

        sample_data['images'] = torch.cat(images, dim=0)
        sample_data['masks'] = torch.cat(masks, dim=0)
        sample_data['bboxes'] = torch.cat(bboxes, dim=0)
        sample_data['Ks'] = torch.cat(Ks, dim=0)

        return sample_data


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

    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(),
            dataset_config['scene_range'])

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
        psnr_monitor = metrics.psnr(rgb_predicted[..., :3] / 2 + 0.5,
                                    target[..., :3] / 2 + 0.5)
        lpips_monitor = module.lpips_net(
            rgb_predicted[..., :3].permute(0, 3, 1, 2),
            target[..., :3].permute(0, 3, 1, 2),
            normalize=False)

    return loss, psnr_monitor, lpips_monitor, rgb_predicted


def evaluate_inversion(obj_idx, it, out_dir, target_img_fid_, target_center_fid, target_bbox_fid, export_sample=False, inception_net=None):
    item = report[it]
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
        camera_flipped=dataset_config['camera_flipped'])
    rgb_predicted, _, acc_predicted, normals_predicted, semantics_predicted, extra_model_outputs = model_to_call(
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
                if dataset_config['white_background']:
                    coords_img += 1 - target_mask.unsqueeze(1)
                demo_img = torch.cat((demo_img, coords_img), dim=3)
            demo_img = torch.cat((demo_img, rgb_predicted_perm), dim=3)
            if normals_predicted is not None:
                demo_img = torch.cat(
                    (demo_img, normals_predicted.permute(0, 3, 1, 2)),
                    dim=3)

            # Move the saving code before
            utils.mkdir(out_dir)
            out_fname = f'demo_obj{obj_idx}_{it}it.png'
            out_path = os.path.join(out_dir, out_fname)
            print('Saving demo output to', out_path)
            torchvision.utils.save_image(demo_img / 2 + 0.5,
                                         out_path,
                                         nrow=1,
                                         padding=0)
    # item['psnr'].append(
    #     metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
    #                  target_perm[:, :3] / 2 + 0.5,
    #                  reduction='none').cpu())
    # item['ssim'].append(
    #     metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
    #                  target_perm[:, :3] / 2 + 0.5,
    #                  reduction='none').cpu())
    # if dataset_config['has_mask']:
    #     item['iou'].append(
    #         metrics.iou(acc_predicted,
    #                     target_perm[:, 3],
    #                     reduction='none').cpu())
    # item['lpips'].append(
    #     loss_fn_lpips(rgb_predicted_perm[:, :3],
    #                   target_perm[:, :3],
    #                   normalize=False).flatten().cpu())
    # if not args.inv_export_demo_sample:
    #     item['inception_activations_front'].append(
    #         torch.FloatTensor(
    #             fid.forward_inception_batch(
    #                 inception_net,
    #                 rgb_predicted_perm[:, :3] / 2 + 0.5)))
    # if not (args.dataset == 'p3d_car' and use_testset):
    #     # Ground-truth poses are not available on P3D Car (test set)
    #     item['rot_error'].append(
    #         pose_utils.rotation_matrix_distance(cam, gt_cam2world_mat))
    #
    # if writer is not None and idx == 0:
    #     if it == checkpoint_steps[0]:
    #         writer.add_images(f'img/ref',
    #                           target_perm[:, :3].cpu() / 2 + 0.5, i)
    #     writer.add_images('img/recon_front',
    #                       rgb_predicted_perm.cpu() / 2 + 0.5, it)
    #     writer.add_images('img/mask_front',
    #                       acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
    #                       it)
    #     if normals_predicted is not None:
    #         writer.add_images(
    #             'img/normals_front',
    #             normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
    #             it)
    #     if semantics_predicted is not None:
    #         writer.add_images(
    #             'img/semantics_front',
    #             (semantics_predicted @ color_palette).cpu().permute(
    #                 0, 3, 1, 2) / 2 + 0.5, it)
    #
    # # Test with random poses
    # rgb_predicted, _, _, normals_predicted, semantics_predicted, _ = model_to_call(
    #     target_tform_cam2world_perm,
    #     target_focal_perm,
    #     target_center_perm,
    #     target_bbox_perm,
    #     z_.detach() * lr_gain_z,
    #     use_ema=True,
    #     compute_normals=args.use_sdf and idx == 0,
    #     compute_semantics=args.attention_values > 0 and idx == 0,
    #     force_no_cam_grad=True,
    #     extra_model_inputs={
    #         k: v.detach() for k, v in extra_model_inputs.items()
    #     },
    # )
    # rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
    #                                                     2).clamp(-1, 1)
    # if export_sample:
    #     with torch.no_grad():
    #         demo_img = torch.cat((demo_img, rgb_predicted_perm), dim=3)
    #         if normals_predicted is not None:
    #             demo_img = torch.cat(
    #                 (demo_img, normals_predicted.permute(0, 3, 1, 2)),
    #                 dim=3)
    #         out_dir = 'outputs'
    #         utils.mkdir(out_dir)
    #         if args.inv_manual_input_path:
    #             out_fname = f'demo_manual_{args.dataset}_{it}it.png'
    #         else:
    #             out_fname = f'sample_{args.dataset}_{it}it.png'
    #         out_path = os.path.join(out_dir, out_fname)
    #         print('Saving demo output to', out_path)
    #         torchvision.utils.save_image(demo_img / 2 + 0.5,
    #                                      out_path,
    #                                      nrow=1,
    #                                      padding=0)
    # if views_per_object > 1:
    #     target_perm_random = target_img_fid_random_.permute(0, 3, 1, 2)
    #     item['psnr_random'].append(
    #         metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
    #                      target_perm_random[:, :3] / 2 + 0.5,
    #                      reduction='none').cpu())
    #     item['ssim_random'].append(
    #         metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
    #                      target_perm_random[:, :3] / 2 + 0.5,
    #                      reduction='none').cpu())
    #     item['lpips_random'].append(
    #         loss_fn_lpips(rgb_predicted_perm[:, :3],
    #                       target_perm_random[:, :3],
    #                       normalize=False).flatten().cpu())
    # if not args.inv_export_demo_sample:
    #     item['inception_activations_random'].append(
    #         torch.FloatTensor(
    #             fid.forward_inception_batch(
    #                 inception_net,
    #                 rgb_predicted_perm[:, :3] / 2 + 0.5)))
    # if writer is not None and idx == 0:
    #     writer.add_images('img/recon_random',
    #                       rgb_predicted_perm.cpu() / 2 + 0.5, it)
    #     writer.add_images('img/mask_random',
    #                       acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
    #                       it)
    #     if normals_predicted is not None:
    #         writer.add_images(
    #             'img/normals_random',
    #             normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
    #             it)
    #     if semantics_predicted is not None:
    #         writer.add_images(
    #             'img/semantics_random',
    #             (semantics_predicted @ color_palette).cpu().permute(
    #                 0, 3, 1, 2) / 2 + 0.5, it)


if __name__ == '__main__':
    # tgt_img_name = 'n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984240912467.jpg'
    # tgt_img_name = 'n008-2018-08-27-11-48-51-0400__CAM_FRONT_RIGHT__1535385099370482.jpg'
    tgt_img_name = 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609912404.jpg'
    gpu = 0

    out_dir = os.path.join('outputs', tgt_img_name[:-4])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load data
    nusc_data_dir = '/mnt/SSD4TB/Datasets/NuScenes/v1.0-mini-full'
    nusc_seg_dir = os.path.join(nusc_data_dir, 'pred_instance')
    nusc_version = 'v1.0-mini'

    nusc_dataset = NuScenesDataset(
        nusc_data_dir,
        nusc_seg_dir,
        nusc_version,
        split='val',
        debug=False,
    )

    # get predicted objects and masks associated with each image
    manual_image = nusc_dataset.get_objects_in_image(tgt_img_name)

    """
        got the minimal viable portion to model to run
    """

    # scene_range = 1.4  # TODO: based on p3d training split, is the trained model overfitted to it?
    # camera_flipped = True  # TODO: based on p3d training split, is the trained model overfitted to it?

    args = arguments.parse_args()

    args.gpus = 1 if args.gpus >= 1 else 0
    args.inv_export_demo_sample = True
    if args.inv_export_demo_sample:
        args.run_inversion = True
    gpu_ids = list(range(args.gpus))

    if args.gpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
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
    # if not args.run_inversion:
    #     utils.mkdir(checkpoint_dir)
    # print('Saving checkpoints to', checkpoint_dir)

    # tensorboard_dir = os.path.join(args.root_path, log_dir, experiment_name)
    report_dir = os.path.join(args.root_path, report_dir)
    # print('Saving tensorboard logs to', tensorboard_dir)
    # if not args.run_inversion:
    #     utils.mkdir(tensorboard_dir)
    # if args.run_inversion:
    print('Saving inversion reports to', report_dir)
    utils.mkdir(report_dir)

    # if args.run_inversion:
    writer = None  # Instantiate later
    # else:
    #     writer = tensorboard.SummaryWriter(tensorboard_dir)

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
    use_encoder = args.use_encoder  # TODO: meaning not use encoder to provide object initial code?
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

    if args.run_inversion:
        # Global config
        use_testset = args.inv_use_testset
        use_pose_regressor = True
        use_latent_regressor = True
        loss_to_use = args.inv_loss
        lr_gain_z = args.inv_gain_z
        inv_no_split = args.inv_no_split
        no_optimize_pose = args.inv_no_optimize_pose

        # if args.inv_manual_input_path:
        #     # Demo inference on manually supplied image
        batch_size = 1
        # else:
        #     batch_size = args.batch_size // 4 * len(gpu_ids)

        # if args.dataset == 'p3d_car' and use_testset:
        #     split_str = 'imagenettest' if args.inv_use_imagenet_testset else 'test'
        # else:
        #     split_str = 'test' if use_testset else 'train'
        # if args.inv_use_separate:
        #     mode_str = '_separate'
        # else:
        #     mode_str = '_joint'
        # if no_optimize_pose:
        #     mode_str += '_nooptpose'
        # else:
        #     mode_str += '_optpose'
        # w_split_str = 'nosplit' if inv_no_split else 'split'
        # cfg_xid = f'_{args.xid}' if len(args.xid) > 0 else ''
        # cfg_string = f'i{cfg_xid}_{split_str}{mode_str}_{loss_to_use}_gain{lr_gain_z}_{w_split_str}'
        # cfg_string += f'_it{resume_from["iteration"]}'
        #
        # print('Config string:', cfg_string)
        #
        # report_dir_effective = os.path.join(report_dir, args.resume_from,
        #                                     cfg_string)
        # print('Saving report in', report_dir_effective)
        # utils.mkdir(report_dir_effective)

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

        # if use_pose_regressor:
        #     focal_guesses = pose_estimation.get_focal_guesses(
        #         train_split.focal_length)
        # TODO: manually set as the p3d training distribution. Is it needed to match the trained model? How to use accurate focal length of the test image?
        focal_guesses = np.asarray([0.71839845,  1.07731938,  1.32769489,  1.59814608,  1.88348041,  2.27928376,
                                    2.82873106,  3.73867059,  5.14416647,  9.12456608, 27.79907417])

        checkpoint_steps = [0, 30]

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
                'rot_error': [],
                'inception_activations_front': [],  # Front view
                'inception_activations_random': [],  # Random view
            } for step in checkpoint_steps
        }

        with torch.no_grad():
            z_avg = model_ema.mapping_network.get_average_w()

        print('Running...')
        num_samples = manual_image['bboxes'].shape[0]
        # deal with each detected object in the image
        for idx, bbox in enumerate(manual_image['bboxes']):
            t1 = time.time()

            # report_checkpoint_path = os.path.join(report_dir_effective,
            #                                       'report_checkpoint.pth')

            target_img = manual_image['images'][idx:idx+1].to(device)
            # target_img = test_split[
            #     target_img_idx].images  # Target for optimization (always cropped)
            target_img_fid_ = target_img  # Target for evaluation (front view -- always cropped)
            # target_tform_cam2world = test_split[target_img_idx].tform_cam2world
            # target_focal = test_split[target_img_idx].focal_length
            target_center = None  # this is for the rendering range in the given image, in pixels. None for full patch
            target_bbox = None  # this is for the rendering range in the given image, in pixels. None for full patch

            target_center_fid = None
            target_bbox_fid = None

            # if use_pose_regressor and 'p3d' in args.dataset:
            #     # Use views from training set (as test pose distribution is not available)
            #     target_center_fid = None
            #     target_bbox_fid = None
            #
            #     target_tform_cam2world_perm = train_eval_split[
            #         target_img_idx_perm].tform_cam2world
            #     target_focal_perm = train_eval_split[
            #         target_img_idx_perm].focal_length
            #     target_center_perm = train_eval_split[
            #         target_img_idx_perm].center
            #     target_bbox_perm = train_eval_split[target_img_idx_perm].bbox
            #
            # gt_cam2world_mat = target_tform_cam2world.clone()
            z_ = z_avg.clone().expand(1, -1, -1).contiguous()

            # TODO: estimated cam2world pose: the object pose in a virtual camera centered at the patch center
            with torch.no_grad():
                coord_regressor_img = target_img[..., :3].permute(0, 3, 1, 2)

                target_coords, target_mask, target_w = coord_regressor.module(
                    coord_regressor_img)

                if use_pose_regressor:
                    assert target_coords is not None
                    estimated_cam2world_mat, estimated_focal, _ = estimate_poses_batch(
                        target_coords, target_mask, focal_guesses)
                    target_tform_cam2world = estimated_cam2world_mat
                    target_focal = estimated_focal
                if use_latent_regressor:
                    assert target_w is not None
                    z_.data[:] = target_w

            if inv_no_split:
                z_ = z_.mean(dim=1, keepdim=True)

            z_ /= lr_gain_z
            z_ = z_.requires_grad_()

            # TODO: this pose representation is for optimization, will it work for actual camera?
            z0_, t2_, s_, R_ = pose_utils.matrix_to_pose(
                target_tform_cam2world,
                target_focal,
                camera_flipped=dataset_config['camera_flipped'])

            if not no_optimize_pose:
                t2_.requires_grad_()
                s_.requires_grad_()
                R_.requires_grad_()
            if z0_ is not None:
                if not no_optimize_pose:
                    z0_.requires_grad_()
                param_list = [z_, z0_, R_, s_, t2_]
                param_names = ['z', 'f', 'R', 's', 't']
            else:
                param_list = [z_, R_, s_, t2_]
                param_names = ['z', 'R', 's', 't']
            if no_optimize_pose:
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

            # if 0 in checkpoint_steps:
            #     evaluate_inversion(0,
            #                        (args.inv_export_demo_sample
            #                         and max(checkpoint_steps) == 0))

            for it in range(niter):
                cam, focal = pose_utils.pose_to_matrix(
                    z0_,
                    t2_,
                    s_,
                    F.normalize(R_, dim=-1),
                    camera_flipped=dataset_config['camera_flipped'])

                # TODO: seems to support perspective camera K as well
                loss, psnr_monitor, lpips_monitor, rgb_predicted = model_to_call(
                    cam,
                    focal,
                    target_center,
                    target_bbox,
                    z_ * lr_gain_z,
                    use_ema=True,
                    ray_multiplier=1 if args.fine_sampling else 4,
                    res_multiplier=1,
                    compute_normals=False and args.use_sdf,
                    force_no_cam_grad=no_optimize_pose,
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

                if args.inv_export_demo_sample:
                    print(it + 1, '/', max(checkpoint_steps))
                if it + 1 in report:
                    evaluate_inversion(idx+1, it + 1, out_dir,
                                       target_img_fid_, target_center_fid, target_bbox_fid,
                                       (args.inv_export_demo_sample and it + 1 == max(checkpoint_steps)))

            t2 = time.time()
            print(
                f'[{idx+1}/{num_samples}] Finished batch in {t2 - t1} s ({(t2 - t1)} s/img)'
            )

            # if args.inv_export_demo_sample:
            #     # Evaluate (and save) only the first batch, then exit
            #     break

            # if idx % 512 == 0:
            #     # Save report checkpoint
            #     with utils.open_file(report_checkpoint_path, 'wb') as f:
            #         torch.save({
            #             'report': report,
            #             'idx': idx,
            #             'test_bs': test_bs,
            #         }, f)
