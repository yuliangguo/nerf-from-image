# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import copy
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io
import cv2
import skimage.io
import glob
import json
import imageio
import matplotlib.pyplot as plt
import pycocotools.mask
from torchvision import transforms
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.nuscenes import NuScenes

from lib import pose_utils
from lib.utils import pts_in_box_3d


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset,
                 split,
                 img_size,
                 root_dir,
                 crop=False,
                 add_mirrored=False):
        self.img_size = img_size
        self.jitter_frac = 0
        self.padding_frac = 0.05

        self.add_mirrored = add_mirrored
        self.crop = crop

        self.debug_disable_mask = False
        self.root_dir = root_dir

        if 'imagenet' in dataset:
            assert split == 'train', 'ImageNet does not have a test split!'

        if split == 'test':
            assert dataset == 'p3d_car'
            p3d_anno_path = os.path.join(root_dir, 'p3d', 'p3d_sfm_image')
            anno_path = os.path.join(p3d_anno_path, 'img_anno', 'car_val.mat')
            val_images = scipy.io.loadmat(anno_path,
                                          struct_as_record=False,
                                          squeeze_me=True)['images']
            self.detections = []
            for img in val_images:
                self.detections.append({
                    'image_path':
                        os.path.join('p3d', 'PASCAL3D+_release1.1', 'Images',
                                     str(img.rel_path).replace('\\', '/')),
                    'bbox':
                        np.array([
                            img.bbox.x1, img.bbox.y1, img.bbox.x2, img.bbox.y2
                        ], float) - 1,
                    'mask':
                        pycocotools.mask.encode(img.mask),
                })

                # Dummy poses
                self.poses = {
                    'f': torch.zeros(len(val_images), 1),
                    't': torch.zeros(len(val_images), 3),
                    'R': torch.zeros(len(val_images), 4),
                }
            return

        if 'imagenet' in dataset:
            path = os.path.join(root_dir, 'imagenet', dataset, 'detections.npy')
            poses_dir = os.path.join(root_dir, 'imagenet', dataset,
                                     'poses_estimated_multitpl_perspective.bin')
        else:
            path = os.path.join(root_dir, 'p3d', dataset, 'detections.npy')
            poses_dir = os.path.join(root_dir, 'p3d', dataset,
                                     'poses_estimated_singletpl_perspective.bin')
        self.detections = np.load(path, allow_pickle=True)

        if split == 'imagenet_test':
            aux_dataset = dataset.replace('p3d', 'imagenet')
            path_aux = os.path.join(root_dir, 'imagenet', aux_dataset,
                                    'detections.npy')
            poses_dir = os.path.join(root_dir, 'imagenet', aux_dataset,
                                     'poses_estimated_multitpl_perspective.bin')
            detections_aux = np.load(path_aux, allow_pickle=True)

            # Build index
            train_set_index = set()
            for item in self.detections:
                img_name = os.path.basename(item['image_path'])
                train_set_index.add(img_name)

            valid_indices = []
            for i, item in enumerate(detections_aux):
                img_name = os.path.basename(item['image_path'])
                if img_name not in train_set_index:
                    valid_indices.append(True)
                else:
                    valid_indices.append(False)
            valid_indices = np.array(valid_indices)
            self.detections = detections_aux

        self.poses = torch.load(poses_dir)
        self.detections = self.detections[self.poses['indices']]  # Pre-filter
        if split == 'imagenet_test':
            valid_indices = valid_indices[self.poses['indices']]
            self.detections = self.detections[valid_indices]
            self.poses = {k: v[valid_indices] for k, v in self.poses.items()}

        # Update camera projection model
        self.poses['f'] = 1 + self.poses['z0'].exp()
        self.poses['t'] = torch.cat((self.poses['t'] / self.poses['s'],
                                     self.poses['f'] / self.poses['s']),
                                    dim=-1)
        del self.poses['z0']
        del self.poses['s']

    @staticmethod
    def quaternion_to_matrix(quaternion):
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        _EPS = np.finfo(float).eps * 4.0
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1. - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1. - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1. - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
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

    @staticmethod
    def perturb_bbox(bbox, pf=0, jf=0):
        pet_bbox = [coord for coord in bbox]
        bwidth = bbox[2] - bbox[0] + 1
        bheight = bbox[3] - bbox[1] + 1

        pet_bbox[0] -= (pf *
                        bwidth) + (1 - 2 * torch.rand(1).item()) * jf * bwidth
        pet_bbox[1] -= (pf *
                        bheight) + (1 - 2 * torch.rand(1).item()) * jf * bheight
        pet_bbox[2] += (pf *
                        bwidth) + (1 - 2 * torch.rand(1).item()) * jf * bwidth
        pet_bbox[3] += (pf *
                        bheight) + (1 - 2 * torch.rand(1).item()) * jf * bheight

        return pet_bbox

    @staticmethod
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

    @staticmethod
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

    def __len__(self):
        if self.add_mirrored:
            return 2 * len(self.detections)
        else:
            return len(self.detections)

    def crop_image(self, img, mask, bbox, sfm_pose):
        # crop image and mask and translate kps
        img = CustomDataset.crop(img, bbox, bgval=1)
        mask = CustomDataset.crop(mask, bbox, bgval=0)
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, sfm_pose

    def scale_image(self, img, mask, sfm_pose, img_size):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = img_size / float(max(bwidth, bheight))
        img_scale, _ = CustomDataset.resize_img(img, scale)
        mask_scale, _ = CustomDataset.resize_img(mask, scale)
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale
        return img_scale, mask_scale, sfm_pose

    def mirror_image(self, img, mask, sfm_pose, bbox=None):
        # Need copy bc torch collate doesnt like neg strides
        img_flip = img[:, ::-1, :].copy()
        mask_flip = mask[:, ::-1].copy()

        sfm_pose[2] *= [1, 1, -1, -1]
        sfm_pose[1][0] *= -1

        if bbox is not None:
            im_w = img.shape[1]
            bbox[0], bbox[2] = im_w - bbox[2], im_w - bbox[0]
            return img_flip, mask_flip, sfm_pose, bbox
        else:
            return img_flip, mask_flip, sfm_pose

    def forward_img(self, idx, manual_image=None):
        if manual_image is None:
            idx_ = idx
            if self.add_mirrored and idx >= len(self.detections):
                idx_ -= len(self.detections)
                mirrored = True
            else:
                mirrored = False
            item = self.detections[idx_]

            img_path_rel = os.path.join(self.root_dir,
                                        item['image_path'].replace('datasets/', ''))
            img_path = img_path_rel
            mask = pycocotools.mask.decode(item['mask'])
            bbox = item['bbox'].flatten()

            img = skimage.io.imread(img_path) / 255.0
            # Some are grayscale:
            if len(img.shape) == 2:
                img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
            mask = np.expand_dims(mask, 2)

            # Sfm pose layout:
            # Focal, Translation xyz, Rot
            sfm_pose = [
                self.poses['f'][idx_].numpy(), self.poses['t'][idx_].numpy(),
                self.poses['R'][idx_].numpy()
            ]
        else:
            img = manual_image['image']
            mask = manual_image['mask']
            bbox = manual_image['bbox']
            mirrored = False
            img_path_rel = ''

            # Dummy pose
            sfm_pose = [
                np.zeros((1,), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                np.zeros((4,), dtype=np.float32),
            ]

        crop = self.crop  # ImageNet / P3D

        if mirrored:
            img, mask, sfm_pose, bbox = self.mirror_image(
                img, mask, sfm_pose, bbox)

        if crop:
            bbox = CustomDataset.perturb_bbox(bbox, pf=self.padding_frac, jf=0)
        else:
            bbox = [0, 0, img.shape[1] - 1, img.shape[0] - 1]

        bbox = CustomDataset.square_bbox(bbox)
        true_resolution = bbox[2] - bbox[0] + 1

        # Compute normalized bbox to return
        max_res = max(img.shape[0], img.shape[1])
        bbox_scaled = bbox.copy()
        if img.shape[0] < img.shape[1]:  # h < w
            bbox_scaled[1] += (max_res - img.shape[0]) / 2
        else:
            bbox_scaled[0] += (max_res - img.shape[1]) / 2

        normalized_bbox_start = np.array([bbox_scaled[0], bbox_scaled[1]
                                         ]) / max_res  # x and y
        normalized_bbox_range = np.array(
            [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1])  # w and h
        assert normalized_bbox_range[0] == normalized_bbox_range[
            1]  # Should be squared
        normalized_bbox_range = normalized_bbox_range / max_res
        normalized_bbox_start = [
            normalized_bbox_start[0],
            1 - normalized_bbox_start[1] - normalized_bbox_range[1]
        ]  # Flip y
        normalized_bbox_start = np.array(normalized_bbox_start) * 2 - 1
        normalized_bbox_range = normalized_bbox_range * 2
        normalized_bbox = np.stack(
            (normalized_bbox_start, normalized_bbox_range), axis=0)

        # important! sfm_pose must not be overwritten -- it is already in the correct reference frame
        img, mask, _ = self.crop_image(img, mask, bbox, copy.deepcopy(sfm_pose))

        # important! sfm_pose must not be overwritten -- it is already in the correct reference frame
        sfm_pose_ref = copy.deepcopy(sfm_pose)
        img_ref, mask_ref, _ = self.scale_image(img.copy(), mask.copy(),
                                                copy.deepcopy(sfm_pose),
                                                self.img_size)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img_ref.shape[:2]

        # Finally transpose the image to 3xHxW
        img_ref = np.transpose(img_ref, (2, 0, 1))

        # Compute transformation matrix
        M = CustomDataset.quaternion_to_matrix(sfm_pose[2])
        M[:3, 3] += sfm_pose[1]
        focal = sfm_pose[0] / 2
        flip = np.eye(4)
        flip[1, 1] = -1
        flip[2, 2] = -1
        M = flip @ M
        M = np.linalg.inv(M)

        class_label = -1
        return img_ref, mask_ref, focal, M, sfm_pose, mirrored, img_path_rel, normalized_bbox, class_label

    def get_paths(self):
        paths = []
        for item in self.detections:
            paths.append(item['image_path'])
        if self.add_flipped:
            paths += paths
        return paths

    def __getitem__(self, index):
        img, mask, focal, M, sfm_pose, mirrored, path, normalized_bbox, class_label = self.forward_img(
            index)
        sfm_pose[0].shape = 1

        # Multiply img with mask
        mask = mask[None, :, :]
        img = img * 2 - 1
        if not self.debug_disable_mask:
            img *= mask
        img = np.concatenate((img, mask), axis=0)

        elem = {
            'img': img.astype(np.float32),
            'normalized_bbox': normalized_bbox.astype(np.float32),
            'focal': focal,
            'pose': M.astype(np.float32),
            'sfm_pose': np.concatenate(sfm_pose).astype(np.float32),
            'mirrored': mirrored,
            'inds': index,
            'path': path,
            'class': class_label,
        }

        return elem


class CUBDataset(CustomDataset):

    def __init__(self,
                 split,
                 img_size,
                 root_dir,
                 crop=False,
                 add_mirrored=False):
        self.img_size = img_size
        self.jitter_frac = 0
        self.padding_frac = 0.05

        self.data_cache_dir = os.path.join(root_dir, 'cub')
        self.data_dir = os.path.join(root_dir, 'cub', 'CUB_200_2011')

        self.img_dir = os.path.join(self.data_dir, 'images')
        self.anno_path = os.path.join(self.data_cache_dir, 'data',
                                      '%s_cub_cleaned.mat' % split)
        self.anno_sfm_path = os.path.join(self.data_cache_dir, 'sfm',
                                          'anno_%s.mat' % split)

        if not os.path.exists(self.anno_path):
            raise ValueError('%s doesnt exist!' % self.anno_path)

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = scipy.io.loadmat(self.anno_path,
                                     struct_as_record=False,
                                     squeeze_me=True)['images']
        self.anno_sfm = scipy.io.loadmat(self.anno_sfm_path,
                                         struct_as_record=False,
                                         squeeze_me=True)['sfm_anno']

        print('%d images' % len(self.anno))

        # Load class labels (if used)
        with open(os.path.join(self.data_dir, 'images.txt'), 'r') as f:
            images = f.readlines()
            images = [x.split(' ') for x in images]
            ids = {k: v.strip() for k, v in images}

        with open(os.path.join(self.data_dir, 'image_class_labels.txt'),
                  'r') as f:
            classes = f.readlines()
            classes = [x.split(' ') for x in classes]
            classes = {k: int(v.strip()) - 1 for k, v in classes}

        self.filename_to_class = {}
        for k, c in classes.items():
            fname = ids[k]
            self.filename_to_class[fname] = c

        self.add_mirrored = add_mirrored
        self.crop = crop

        self.debug_disable_mask = False

    def __len__(self):
        if self.add_mirrored:
            return 2 * len(self.anno)
        else:
            return len(self.anno)

    def get_paths(self):
        paths = []
        for index, data in enumerate(self.anno):
            img_path_rel = str(data.rel_path).replace('\\', '/')
            paths.append(img_path_rel)
        return paths

    def normalize_kp(self, sfm_pose, img_h, img_w):
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        return sfm_pose

    def forward_img(self, idx, manual_image=None):
        if manual_image is not None:
            return super().forward_img(idx, manual_image)

        idx_ = idx
        if self.add_mirrored and idx >= len(self.anno):
            idx_ -= len(self.anno)
            mirrored = True
        else:
            mirrored = False

        data = self.anno[idx_]
        data_sfm = self.anno_sfm[idx_]

        sfm_pose = [
            np.copy(data_sfm.scale),
            np.copy(data_sfm.trans),
            np.copy(data_sfm.rot)
        ]

        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = pose_utils.matrix_to_quaternion(sfm_rot)

        img_path = os.path.join(self.img_dir,
                                str(data.rel_path)).replace('\\', '/')
        img_path_rel = str(data.rel_path).replace('\\', '/')
        img = skimage.io.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        class_label = self.filename_to_class[img_path_rel]

        crop = self.crop  # CUB

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float) - 1

        # Perturb bbox
        if crop:
            bbox = CustomDataset.perturb_bbox(bbox, pf=self.padding_frac, jf=0)
        else:
            bbox = [0, 0, img.shape[1] - 1, img.shape[0] - 1]

        bbox = CustomDataset.square_bbox(bbox)
        true_resolution = bbox[2] - bbox[0] + 1

        # crop image around bbox, translate kps
        img, mask, sfm_pose = self.crop_image(img, mask, bbox, sfm_pose)

        # scale image, and mask. And scale kps.
        img_ref, mask_ref, sfm_pose_ref = self.scale_image(
            img.copy(), mask.copy(), copy.deepcopy(sfm_pose), self.img_size)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img_ref.shape[:2]
        sfm_pose_ref = self.normalize_kp(sfm_pose_ref, img_h, img_w)

        if mirrored:
            img_ref, mask_ref, sfm_pose_ref = self.mirror_image(
                img_ref, mask_ref, sfm_pose_ref)

        # Finally transpose the image to 3xHxW
        img_ref = np.transpose(img_ref, (2, 0, 1))

        # Compute transformation matrix
        M = CustomDataset.quaternion_to_matrix(sfm_pose_ref[2])
        M[:3, :3] *= sfm_pose_ref[0]  # Scale
        M[3, 3] *= sfm_pose_ref[0]  # Scale
        M[:2, 3] += sfm_pose_ref[1]
        M[2, 3] += 10  # Fixed offset from center (near clipping plane)
        M[:3, 3] *= sfm_pose_ref[0]
        flip = np.eye(4)
        flip[1, 1] = -1
        flip[2, 2] = -1
        M = flip @ M
        M = np.linalg.inv(M)

        normalized_bbox = np.zeros(1)  # Not needed
        focal = np.zeros(1)  # Not needed
        return img_ref, mask_ref, focal, M, sfm_pose_ref, mirrored, img_path_rel, normalized_bbox, class_label


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
            self,
            path,
            stage='train',
            image_size=(128, 128),
            world_scale=1.0,
            limit=None,
    ):
        super().__init__()
        self.base_path = path + '_' + stage
        self.dataset_name = os.path.basename(path)

        print('Loading SRN dataset', self.base_path, 'name:', self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = 'chair' in self.dataset_name
        if is_chair and stage == 'train':
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, 'chairs_2.0_train')
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, '*', 'intrinsics.txt')))
        self.image_to_tensor = SRNDataset.get_image_to_tensor_balanced()
        self.mask_to_tensor = SRNDataset.get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False
        self.limit = limit

    @staticmethod
    def get_image_to_tensor_balanced(image_size=0):
        ops = []
        if image_size > 0:
            ops.append(transforms.Resize(image_size))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transforms.Compose(ops)

    @staticmethod
    def get_mask_to_tensor():
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.0,), (1.0,))])

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, 'rgb', '*')))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, 'pose', '*')))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, 'r') as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []

        if self.limit is not None:
            indices = np.random.choice(len(rgb_paths),
                                       size=(self.limit,),
                                       replace=False)
            rgb_paths = [rgb_paths[i] for i in indices]
            pose_paths = [pose_paths[i] for i in indices]

        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).any(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4))
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale

            all_imgs = F.interpolate(all_imgs,
                                     size=self.image_size,
                                     mode='area')
            all_masks = F.interpolate(all_masks,
                                      size=self.image_size,
                                      mode='area')

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32) / self.image_size[0]

        result = {
            'img_id':
                index,
            'focal':
                focal,
            'c':
                torch.tensor([cx, cy], dtype=torch.float32) /
                self.image_size[0],
            'images':
                all_imgs,
            'masks':
                all_masks,
            'poses':
                all_poses,
        }
        return result


class CARLADataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, image_size, upscale=False):
        self.img_paths = sorted(glob.glob(os.path.join(dataset_path, '*.png')))
        print(len(self.img_paths), 'images')

        self.image_size = image_size
        self.upscale = 2 if upscale else 1

        # Load poses
        poses = []
        for img_path in self.img_paths:
            pose_path = os.path.join(
                dataset_path, 'carla_poses',
                os.path.basename(img_path).replace('.png', '_extrinsics.npy'))
            poses.append(np.load(pose_path))
        self.poses = np.zeros((len(poses), 4, 4), dtype=np.float32)
        self.poses[:, :3] = np.stack(poses, axis=0)
        self.poses[:, 3, 3] = 1

        intrinsics = np.load(
            os.path.join(dataset_path, 'carla_poses', 'intrinsics.npy'))
        self.c = intrinsics[0, 0, :2, 2].astype(np.float32)
        self.focal = intrinsics[0, 0, 0, 0].astype(np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        img = imageio.imread(rgb_path)[..., :3]
        original_res = img.shape[0]
        img = img.astype(np.float32) / 255 * 2 - 1
        img = torch.FloatTensor(img).permute(2, 0, 1)
        img = F.interpolate(img.unsqueeze(0),
                            size=self.image_size * self.upscale,
                            mode='area')
        return {
            'focal': torch.FloatTensor([self.focal.item()]) / original_res,
            'c': torch.FloatTensor(self.c) / original_res,
            'image': img.squeeze(0),
            'pose': torch.FloatTensor(self.poses[idx]),
        }


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 nusc_data_dir,
                 nusc_seg_dir,
                 nusc_version,
                 split='val',
                 img_size=128,
                 debug=False,
                 external_pose_file=None,
                 ):
        self.nusc_cat = 'vehicle.car'
        self.seg_cat = 'car'
        self.nusc_data_dir = nusc_data_dir
        self.nusc_seg_dir = nusc_seg_dir
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_data_dir, verbose=True)
        self.img_size = img_size

        # load pre-prepared qualified sample indices
        subset_index_file = 'data/nusc.' + nusc_version + '.' + split + '.' + self.nusc_cat + '.json'
        nusc_subset = json.load(open(subset_index_file))
        self.all_valid_samples = nusc_subset['all_valid_samples']
        self.anntokens_per_ins = nusc_subset['anntokens_per_ins']
        self.instoken_per_ann = nusc_subset['instoken_per_ann']
        self.sample_attr = nusc_subset['sample_attr']
        self.lenids = len(self.all_valid_samples)
        print('Loaded existing index file for valid samples.')

        print('Preparing camera data dictionary for fast retrival given image name')
        self.cam_data_dict = {}
        for sample in self.nusc.sample_data:
            if 'CAM' in sample['channel']:
                self.cam_data_dict[os.path.basename(sample['filename'])] = sample

        self.out_gt_depth = True
        self.pred_box2d = False
        self.debug = debug

        if external_pose_file is not None:
            saved_results = torch.load(external_pose_file, map_location=torch.device('cpu'))
            self.optimized_poses = saved_results['optimized_poses']

    def get_mask_occ_from_ins(self, masks, tgt_ins_id):
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

    def __len__(self):
        return self.lenids

    def __getitem__(self, idx):
        sample_data = {}
        anntoken, cam = self.all_valid_samples[idx]
        if self.debug:
            print(f'anntoken: {anntoken}')

        # For each annotation (one annotation per timestamp) get all the sensors
        sample_ann = self.nusc.get('sample_annotation', anntoken)
        sample_record = self.nusc.get('sample', sample_ann['sample_token'])

        # Figure out which camera the object is fully visible in (this may return nothing).
        if self.debug:
            print(f'     {cam}')
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam],
                                                                       box_vis_level=BoxVisibility.ALL,
                                                                       selected_anntokens=[anntoken])
        # Plot CAMERA view.
        img = imageio.imread(data_path)

        # box here is in sensor coordinate system
        box = boxes[0]
        # compute the camera pose in object frame, make sure dataset and model definitions consistent
        obj_center = box.center
        obj_orientation = box.orientation.rotation_matrix
        obj_pose = np.concatenate([obj_orientation, np.expand_dims(obj_center, -1)], axis=1)
        # Compute camera pose in object frame = c2o transformation matrix
        # Recall that object pose in camera frame = o2c transformation matrix
        R_c2o = obj_orientation.transpose()
        t_c2o = - R_c2o @ np.expand_dims(obj_center, -1)
        cam_pose = np.concatenate([R_c2o, t_c2o], axis=1)

        # find the valid instance given 2d box projection
        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
        min_x = np.min(corners[0, :])
        max_x = np.max(corners[0, :])
        min_y = np.min(corners[1, :])
        max_y = np.max(corners[1, :])
        box_2d = [min_x, min_y, max_x, max_y]

        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        for box_id in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(data_path)[:-4] + f'_{box_id}.png')
            mask = imageio.imread(mask_file)
            ins_masks.append(mask)
        if len(ins_masks) == 0:
            mask_occ = None
        else:
            tgt_ins_id = self.sample_attr[anntoken][cam]['seg_id']
            mask_occ = self.get_mask_occ_from_ins(ins_masks, tgt_ins_id)
            if self.pred_box2d:
                box_2d = preds['boxes'][tgt_ins_id]
                # enlarge pred_box
                # box_2d = roi_resize(box_2d, ratio=self.box2d_rz_ratio)
        # lidar_cnt = self.sample_attr[anntoken][cam]['lidar_cnt']

        # Prepare gt depth map using lidar points
        pointsensor_token = sample_record['data']['LIDAR_TOP']
        camtoken = sample_record['data'][cam]
        # lidarseg_idx = self.nusc.lidarseg_name2idx_mapping[self.nusc_cat]

        # Only the lider points within the target camera image belong to the target class is returned.
        lidar_pts_im, lider_pts_depth, _ = self.nusc.explorer.map_pointcloud_to_image(pointsensor_token, camtoken,
                                                                                      render_intensity=False,
                                                                                      show_lidarseg=False,
                                                                                      filter_lidarseg_labels=None,
                                                                                      lidarseg_preds_bin_path=None,
                                                                                      show_panoptic=False)

        lidar_pts_cam = np.matmul(np.linalg.inv(camera_intrinsic), lidar_pts_im) * lider_pts_depth
        pts_ann_indices = pts_in_box_3d(lidar_pts_cam, box.corners(), keep_top_portion=0.9)
        lidar_pts_im_ann = lidar_pts_im[:, pts_ann_indices]
        lider_pts_depth_ann = lider_pts_depth[pts_ann_indices]

        depth_map = np.zeros(img.shape[:2]).astype(np.float32)
        depth_map[
            lidar_pts_im_ann[1, :].astype(np.int32), lidar_pts_im_ann[0, :].astype(np.int32)] = lider_pts_depth_ann
        sample_data['depth_maps'] = torch.from_numpy(depth_map.astype(np.float32))

        obj_pose_ext = self.optimized_poses[anntoken][cam]
        # ATTENTION: prepare batch data including ray based samples can further improve efficiency,
        # but lower flexible for training considering different crop sizes

        sample_data['imgs'] = torch.from_numpy(img.astype(np.float32) / 255.)
        sample_data['masks_occ'] = torch.from_numpy(mask_occ.astype(np.float32))
        sample_data['rois'] = torch.from_numpy(np.asarray(box_2d).astype(np.int32))
        sample_data['cam_intrinsics'] = torch.from_numpy(camera_intrinsic.astype(np.float32))
        sample_data['cam_poses'] = torch.from_numpy(np.asarray(cam_pose).astype(np.float32))
        sample_data['obj_poses'] = torch.from_numpy(np.asarray(obj_pose).astype(np.float32))
        sample_data['obj_poses_ext'] = obj_pose_ext
        sample_data['instoken'] = self.instoken_per_ann[anntoken]
        sample_data['anntoken'] = anntoken
        sample_data['cam_ids'] = cam
        wlh = self.nusc.get('sample_annotation', anntoken)['size']
        sample_data['wlh'] = torch.tensor(wlh, dtype=torch.float32)

        """
            Prepare cropped date for BootInv
        """
        bbox = CustomDataset.square_bbox(box_2d)
        K = camera_intrinsic.copy()

        # important! sfm_pose must not be overwritten -- it is already in the correct reference frame
        img = img.astype(np.float32).copy() / 255.
        img = CustomDataset.crop(img, bbox, bgval=1)
        mask = (mask_occ > 0).astype(np.float32)[:, :, None]
        mask = CustomDataset.crop(mask, bbox, bgval=0)
        depth_map = depth_map.copy()[:, :, None]
        depth_map = CustomDataset.crop(depth_map, bbox, bgval=-1)
        K[0, 2] -= (bbox[0] + bbox[2])/2
        K[1, 2] -= (bbox[1] + bbox[3])/2

        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img, _ = CustomDataset.resize_img(img, scale)
        mask, _ = CustomDataset.resize_img(mask, scale)
        # resize sparse depth using nearest rather than interpolation
        depth_map = cv2.resize(depth_map, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # K[:2, :] *= scale
        K[0, :] /= float(max(bwidth, bheight))
        K[1, :] /= float(max(bwidth, bheight))

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        mask = mask[None, :, :]
        img = img * 2 - 1
        img *= mask
        img = torch.FloatTensor(img).permute(1, 2, 0)

        sample_data['img_batch'] = img
        sample_data['mask_batch'] = torch.FloatTensor(mask.squeeze())
        sample_data['depth_batch'] = torch.FloatTensor(depth_map)
        sample_data['bbox_batch'] = torch.FloatTensor(bbox)
        sample_data['K_batch'] = torch.FloatTensor(K)

        # if self.debug:
        #     print(
        #         f'        tgt instance id: {tgt_ins_id}, '
        #         f'lidar pts cnt: {lidar_cnt} ')
        #
        #     camtoken = sample_record['data'][cam]
        #     fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        #
        #     # draw object box on the image
        #     img2 = np.copy(img)
        #     corners_3d = corners_of_box(obj_pose, wlh, is_kitti=False)
        #     pred_uv = view_points(
        #         corners_3d,
        #         camera_intrinsic, normalize=True)
        #     c = np.array([0, 255, 0]).astype(np.float)
        #     img2 = render_box(img2, pred_uv, colors=(c, c, c))
        #     if self.add_pose_err > 0:
        #         corners_3d_w_err = corners_of_box(obj_pose_w_err, wlh, is_kitti=False)
        #         # corners_3d_w_err = np.array(objects_pred['corners_3d'][asso_obx_id]).T
        #         pred_uv_w_err = view_points(
        #             corners_3d_w_err,
        #             camera_intrinsic, normalize=True)
        #         c = np.array([255, 0, 0]).astype(np.float)
        #         img2 = render_box(img2, pred_uv_w_err, colors=(c, c, c))
        #     axes[0].imshow(img2)
        #     axes[0].set_title(self.nusc.get('sample_data', camtoken)['channel'])
        #     axes[0].axis('off')
        #     axes[0].set_aspect('equal')
        #     # c = np.array(self.nusc.colormap[box.name]) / 255.0
        #     # box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c))
        #
        #     if self.seg_type == 'panoptic':
        #         seg_vis = pan2ins_vis(pan_label, name2label[self.seg_cat][2], self.divisor)
        #     elif self.seg_type == 'instance':
        #         seg_vis = ins2vis(ins_masks)
        #     axes[1].imshow(seg_vis)
        #     axes[1].set_title('pred instance')
        #     axes[1].axis('off')
        #     axes[1].set_aspect('equal')
        #     # c = np.array(nusc.colormap[box.name]) / 255.0
        #     min_x, min_y, max_x, max_y = box_2d
        #     rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
        #                              linewidth=2, edgecolor='y', facecolor='none')
        #     axes[1].add_patch(rect)
        #
        #     axes[0].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)
        #     axes[1].scatter(lidar_pts_im_ann[0, :], lidar_pts_im_ann[1, :], c=lider_pts_depth_ann, s=5)
        #
        #     self.nusc.render_annotation(anntoken, margin=30, box_vis_level=BoxVisibility.ALL)
        #     plt.tight_layout()
        #     plt.show()
        #     print(f"        Lidar pts in target segment: {lidar_cnt}")
        #     # Nusc claimed pixel ratio visible from 6 cameras, seem not very reliable since no GT amodel segmentation
        #     visibility_token = sample_ann['visibility_token']
        #     print("        Visibility: {}".format(self.nusc.get('visibility', visibility_token)))

        return sample_data

    def get_objects_in_image(self, filename):
        """
            Output mask-rcnn masks and boxes per image
        """
        if filename not in self.cam_data_dict.keys():
            print(f'Target image file {filename} does not contain valid annotations')
            return None

        cam_data = self.cam_data_dict[filename]
        # load image, 2D boxes and masks, only need to get K from nusc
        impath, _, camera_intrinsic = self.nusc.get_sample_data(cam_data['token'], box_vis_level=BoxVisibility.ANY)

        # load image
        # img_org = Image.open(impath)
        # img_org = np.asarray(img_org)
        img_org = imageio.imread(impath)

        # load mask-rcnn predicted instance masks and 2D boxes
        cam = cam_data['channel']
        json_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + '.json')
        preds = json.load(open(json_file))
        ins_masks = []
        rois = []
        for ii in range(0, len(preds['boxes'])):
            mask_file = os.path.join(self.nusc_seg_dir, cam, os.path.basename(impath)[:-4] + f'_{ii}.png')
            # mask = np.asarray(Image.open(mask_file))
            mask = imageio.imread(mask_file)
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
            mask_occ = self.get_mask_occ_from_ins(ins_masks, ii)
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
            bbox = CustomDataset.square_bbox(bbox)
            max_res = max(img_org.shape[0], img_org.shape[1])

            K = camera_intrinsic.copy()
            # important! sfm_pose must not be overwritten -- it is already in the correct reference frame
            img = img_org.astype(np.float32).copy()/255.
            img = CustomDataset.crop(img, bbox, bgval=1)
            mask = ins_masks[ii].copy()[:, :, None]/255
            mask = CustomDataset.crop(mask, bbox, bgval=0)
            K[0, 2] -= bbox[0]
            K[1, 2] -= bbox[1]

            # Scale image so largest bbox size is img_size
            bwidth = np.shape(img)[0]
            bheight = np.shape(img)[1]
            scale = self.img_size / float(max(bwidth, bheight))
            img, _ = CustomDataset.resize_img(img, scale)
            mask, _ = CustomDataset.resize_img(mask, scale)
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

        # TODO: need to sync with the right setup updated in __getitem__
        sample_data['images'] = torch.cat(images, dim=0)
        sample_data['masks'] = torch.cat(masks, dim=0)
        sample_data['bboxes'] = torch.cat(bboxes, dim=0)
        sample_data['Ks'] = torch.cat(Ks, dim=0)

        return sample_data