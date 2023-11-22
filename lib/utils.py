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

import pathlib
import contextlib
import os
import io
import torch
import numpy as np


@contextlib.contextmanager
def open_file(path, mode):
    """Opens a file in read or write mode. Supports Google Cloud Storage."""
    if path.startswith('gs://'):
        from tensorflow.io import gfile  # pylint: disable=C0415
        with gfile.GFile(path, mode) as f:
            try:
                yield io.BytesIO(f.read()) if 'r' in mode else f
            finally:
                pass
    else:
        with open(path, mode) as f:  # pylint: disable=W1514
            try:
                yield f
            finally:
                pass


def mkdir(path):
    """Creates a directory if non-existent. Supports Google Cloud Storage."""
    if path.startswith('gs://'):
        from tensorflow.io import gfile  # pylint: disable=C0415
        gfile.makedirs(path)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    """Checks if a file exists. Supports Google Cloud Storage."""
    if path.startswith('gs://'):
        from tensorflow.io import gfile  # pylint: disable=C0415
        return gfile.exists(path)
    else:
        return os.path.isfile(path)


def get_color_palette(n):
    """Returns a random color palette for the color mapping visualization."""
    if n == 0:
        return None

    color_palette = torch.FloatTensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 0],
        [1, 0.5, 0],
        [0.5, 1, 0],
        [0, 0.5, 1],
    ]) * 2 - 1

    if n == len(color_palette):
        return color_palette
    elif n < len(color_palette):
        return color_palette[:n]
    else:
        nrep = (len(color_palette) + n - 1) // len(color_palette)
        color_palette = color_palette.repeat(nrep, 1)
        return color_palette[:n]


def save_random_state(data_sampler, rng, gpu_ids):
    rng_state = {}
    rng_state['np_global_state'] = np.random.get_state()
    rng_state['np_rng_state'] = rng.get_state()
    rng_state['torch_cpu_state'] = torch.get_rng_state()
    torch_cuda_state = []
    for device_id in gpu_ids:
        with torch.cuda.device(device_id):
            torch_cuda_state.append(torch.cuda.get_rng_state())
    rng_state['torch_cuda_state'] = torch_cuda_state
    rng_state['data_sampler_state'] = data_sampler.get_state()
    return rng_state


def restore_random_state(rng_state, data_sampler, rng, gpu_ids):
    np.random.set_state(rng_state['np_global_state'])
    rng.set_state(rng_state['np_rng_state'])
    torch.set_rng_state(rng_state['torch_cpu_state'])

    if len(rng_state['torch_cuda_state']) != len(gpu_ids):
        print('Warning: original checkpoint was trained with a different '
              'number of GPUs. Cannot fully restore random state.')
    for device_id, random_state_cuda in zip(gpu_ids,
                                            rng_state['torch_cuda_state']):
        with torch.cuda.device(device_id):
            torch.cuda.set_rng_state(random_state_cuda)
    data_sampler.set_state(rng_state['data_sampler_state'], rng)


def load_manual_image(path_or_url, coco_class_id):
    # On-demand imports
    import urllib
    import PIL
    import detectron2
    import detectron2.config
    import detectron2.model_zoo
    import detectron2.engine

    if path_or_url.startswith('http'):
        with urllib.request.urlopen(path_or_url) as response:
            demo_input_img = PIL.Image.open(io.BytesIO(response.read()))
    else:
        demo_input_img = PIL.Image.open(path_or_url)
    demo_input_img = np.array(demo_input_img)
    assert len(demo_input_img.shape) == 3
    assert demo_input_img.shape[-1] in [3, 4] # RGB(A)
    demo_input_img = demo_input_img[:, :, :3]

    # In the paper we use PointRend to extract masks,
    # but for a demo a simpler model is also fine.
    # cfg_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    cfg_file = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Detection threshold
    cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(cfg_file)
    predictor = detectron2.engine.DefaultPredictor(cfg)

    # Detectron expects BGR format
    outputs = predictor(demo_input_img[:, :, ::-1])['instances']
    outputs = outputs[outputs.pred_classes == coco_class_id]
    if len(outputs) == 0:
        raise RuntimeError('Could not detect any object in the provided image')
    
    # Extract largest detected object
    outputs = outputs[outputs.pred_masks.sum(dim=[1, 2]).argmax().item()]

    manual_image = {
        'image': demo_input_img.astype(np.float32) / 255,
        'mask': outputs.pred_masks[0].cpu().float().unsqueeze(-1),
        'bbox': outputs.pred_boxes.tensor[0].cpu().tolist(),
    }
    return manual_image


class EndlessSampler:

    def __init__(self, dataset_size, rng):
        self.ptr = 0
        self.dataset_size = dataset_size
        self.shuffled_indices = None
        self.rng = rng

    def get_state(self):
        return self.ptr, self.shuffled_indices

    def set_state(self, state, rng):
        self.rng = rng
        self.ptr, self.shuffled_indices = state
        assert (self.shuffled_indices is None or
                len(self.shuffled_indices) == self.dataset_size)

    def _yield_batch(self, batch_size):
        for _ in range(batch_size):
            if self.shuffled_indices is None:
                self.shuffled_indices = self.rng.permutation(self.dataset_size)
            next_elem = self.shuffled_indices[self.ptr]
            self.ptr += 1
            if self.ptr == self.dataset_size:
                self.ptr = 0
                self.shuffled_indices = None
            yield next_elem

    def __call__(self, batch_size):
        return list(self._yield_batch(batch_size))


def pts_in_box_3d(pts_3d, corners_3d, keep_top_portion=1.0):
    """
        Extract lidar points associated within the annotation 3D box
        Both pts_3d and corners_3d (nusc def) live in the same coordinate frame
        keep_ratio is the top portion to be kept

        return the indices
    """
    v1 = (corners_3d[:, 1:2] - corners_3d[:, 0:1])
    v2 = (corners_3d[:, 3:4] - corners_3d[:, 0:1]) * keep_top_portion
    v3 = (corners_3d[:, 4:5] - corners_3d[:, 0:1])
    v_test = pts_3d - corners_3d[:, 0:1]

    proj_1 = np.matmul(v1.T, v_test)
    proj_2 = np.matmul(v2.T, v_test)
    proj_3 = np.matmul(v3.T, v_test)

    subset1 = np.logical_and(proj_1 > 0,  proj_1 < np.matmul(v1.T, v1))
    subset2 = np.logical_and(proj_2 > 0,  proj_2 < np.matmul(v2.T, v2))
    subset3 = np.logical_and(proj_3 > 0,  proj_3 < np.matmul(v3.T, v3))

    subset_final = np.logical_and(subset1, np.logical_and(subset2, subset3))
    return np.squeeze(subset_final)


def corners_of_box(obj_pose, wlh, is_kitti=False):
    """
            Modified from NUSC

    Returns the bounding box corners.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = wlh

    if is_kitti:
        # the order (identity) need to follow nusc
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = h / 2 * np.array([-2, -2, 0, 0, -2, -2, 0, 0])
        z_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    else:
        # 3D bounding box corners. (Convention: x forward, y left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(obj_pose[:, :3], corners)

    # Translate
    x, y, z = obj_pose[:, 3]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def obj_pose_kitti2nusc(obj_pose_src, obj_h):
    """
        Operate at batch level
    """
    pose_R = obj_pose_src[:, :, :3]
    pose_T = obj_pose_src[:, :, 3:]
    pose_T[:, 1, 0] -= (obj_h/2)
    R_x = np.array([[1., 0., 0.],
                    [0., 0., -1.],
                    [0., 1., 0.]]).astype(np.float32)
    R_x = torch.from_numpy(R_x).unsqueeze(0).repeat(obj_pose_src.shape[0], 1, 1)
    pose_R = torch.matmul(pose_R, R_x)
    return torch.cat([pose_R, pose_T], dim=-1)


def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): random seed
    """
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
