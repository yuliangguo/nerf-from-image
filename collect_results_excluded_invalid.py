import torch
import numpy as np

if __name__ == '__main__':
    result_file = 'outputs/waymo_init_external_opt_pose_True_2024_01_30_17/report_checkpoint.pth'

    saved_report = torch.load(result_file)

    # subset v0
    num_samples = 1225
    indices2exclude = [60, 217, 220, 260, 282, 309, 320, 327, 346, 555, 594, 601, 633, 651, 671, 740, 780, 843, 851, 869, 879, 894, 958, 1013, 1033, 1036, 1066, 1081, 1151, 1152, 1205, 1221]
    # num_samples = 699
    # indices2exclude = [26, 32, 76, 79, 82, 83, 91, 115, 140, 142, 153, 154, 171, 173, 179, 180, 181, 191, 194, 219, 258, 266, 269, 272, 286, 295, 296, 305, 314, 325, 328, 329, 330, 332, 334, 341, 351, 360, 362, 367, 409, 440, 444, 462, 466, 478, 479, 483, 484, 485, 496, 500, 501, 516, 539, 555, 560, 562, 567, 581, 603, 604, 610, 612, 616, 624, 628, 632, 638, 650, 653, 660, 668, 669]
    indices2include = [idx for idx in range(num_samples) if idx not in indices2exclude]

    psnr_all = [
        torch.tensor(saved_report['report'][0]['psnr'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][20]['psnr'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][50]['psnr'])[indices2include].mean().item(),
    ]
    # psnr_all = np.array(psnr_all)

    rot_err_all = [
        torch.tensor(saved_report['report'][0]['rot_error'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][20]['rot_error'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][50]['rot_error'])[indices2include].mean().item(),
    ]
    # rot_err_all = np.array(rot_err_all)

    trans_err_all = [
        torch.tensor(saved_report['report'][0]['trans_error'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][20]['trans_error'])[indices2include].mean().item(),
        torch.tensor(saved_report['report'][50]['trans_error'])[indices2include].mean().item(),
    ]
    # trans_all_all = np.array(trans_all_all)

    print('evaluation at iter 0, 20, 50')
    print(f'psnr: {psnr_all}')
    print(f'rot error: {rot_err_all}')
    print(f'trans error: {trans_err_all}')


