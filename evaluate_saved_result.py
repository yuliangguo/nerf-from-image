import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default='outputs/waymo_init_external_opt_pose_True_2024_01_30_17/report_checkpoint.pth')
    args = parser.parse_args()

    # result_file = 'outputs/waymo_init_external_opt_pose_True_2024_01_30_17/report_checkpoint.pth'
    saved_report = torch.load(args.result_file)

    psnr_all = [
        torch.tensor(saved_report['report'][0]['psnr']).mean().item(),
        torch.tensor(saved_report['report'][20]['psnr']).mean().item(),
        torch.tensor(saved_report['report'][50]['psnr']).mean().item(),
    ]
    # psnr_all = np.array(psnr_all)

    rot_err_all = [
        torch.tensor(saved_report['report'][0]['rot_error']).mean().item(),
        torch.tensor(saved_report['report'][20]['rot_error']).mean().item(),
        torch.tensor(saved_report['report'][50]['rot_error']).mean().item(),
    ]
    # rot_err_all = np.array(rot_err_all)

    trans_err_all = [
        torch.tensor(saved_report['report'][0]['trans_error']).mean().item(),
        torch.tensor(saved_report['report'][20]['trans_error']).mean().item(),
        torch.tensor(saved_report['report'][50]['trans_error']).mean().item(),
    ]
    # trans_all_all = np.array(trans_all_all)

    print('evaluation at iter 0, 20, 50')
    print(f'psnr: {psnr_all}')
    print(f'rot error: {rot_err_all}')
    print(f'trans error: {trans_err_all}')
