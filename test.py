"""测试融合网络"""
import argparse
import os
import random

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from data_loader.common import YCrCb2RGB, RGB2YCrCb, clamp
from net import LE2Fusion

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model_LE2Fusion import LE2
import torch

# torch.cuda.set_device(1)

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LE2Fusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='/data/Disk_A/yongbiao/USERPROG/LE2Fusion/test_data/MSRS-main',
                        help='path to dataset (default: imagenet)')# 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='./results/LE2Fusion_MSRS-main')# 融合结果存放位置
    parser.add_argument('--vi_weight_save_path', default='result_train-20/LE2Fusion/weight_vi')# 融合结果权重图存放位置
    parser.add_argument('--ir_weight_save_path', default='result_train-20/LE2Fusion/weight_ir')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='fusion_model_epoch_4.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    # init_seeds(args.seed)

    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # if not os.path.exists(args.ir_weight_save_path):
    #     os.makedirs(args.ir_weight_save_path)
    # if not os.path.exists(args.vi_weight_save_path):
    #     os.makedirs(args.vi_weight_save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = LE2Fusion()
        model = model.cuda()
        # model = model.to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.fusion_pretrained,map_location=torch.device('cpu')))
        print(model)
        model.eval()
        cls = LE2(3)
        cls.cuda()
        cls.eval()
        print(cls)
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_image, vis_y_image, cb, cr, inf_image, name in test_tqdm:
                vis_image = vis_image.cuda()
                vis_y_image = vis_y_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                fused_image = model(vis_y_image, inf_image,vis_image)

                fused_image = clamp(fused_image)


########################################测试中输出权重图#####################
                out = cls(vis_image)
                out_origin = torch.tensor_split(out, 2, dim=1)
                out_ir = out_origin[0]
                out_vi = out_origin[1]

                ir_e = torch.exp(out_ir)
                vi_e = torch.exp(out_vi)

                ir_weight = ir_e / (ir_e + vi_e)
                vi_weight = vi_e / (ir_e + vi_e)
                #保存灰度图
                # unloader = transforms.ToPILImage()
                # image = (ir_weight[0]).cpu().clone()  # clone the tensor
                # image = image.squeeze(0)  # remove the fake batch dimension
                # image = unloader(image)
                # image.save(f'{args.ir_weight_save_path}/{name[0]}')
                # #
                # unloader1 = transforms.ToPILImage()
                # image1 = (vi_weight[0]).cpu().clone()  # clone the tensor
                # image1 = image1.squeeze(0)  # remove the fake batch dimension
                # image1 = unloader1(image1)
                # image1.save(f'{args.vi_weight_save_path}/{name[0]}')

                #保存rgb图
                # rgb_inf_weight = YCrCb2RGB(ir_weight[0], cb[0], cr[0])
                # rgb_inf_weight = transforms.ToPILImage()(rgb_inf_weight)
                # rgb_inf_weight.save(f'{args.ir_weight_save_path}/{name[0]}')
                #
                # rgb_vis_weight = YCrCb2RGB(vi_weight[0], cb[0], cr[0])
                # rgb_vis_weight = transforms.ToPILImage()(rgb_vis_weight)
                # rgb_vis_weight.save(f'{args.vi_weight_save_path}/{name[0]}')





    ########################################测试中输出权重图#####################

                #格式转换，因为tensor不能直接保存成图片
                rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                # fused_image = torch.sum(fused_image, dim=0)
                # fused_image = torch.sum(fused_image, dim=0)
                # fused_image = fused_image.detach().cpu().numpy()
                # import imageio
                # imageio.imsave(f'{args.save_path}/{name[0]}',fused_image)
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')


