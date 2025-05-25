# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/25 13:00
@File ：train_fusion_model.py
@IDE ：PyCharm
@Function ：训练融合网络
"""
import argparse
import os
import random
import numpy as np
import torch

import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_msssim
from kornia.losses import ssim_loss
from data_loader.msrs_2 import MSRS_data
from data_loader.pixel_intensity_loss import pixel_intensity
from model_LE2Fusion import LE2
from data_loader.common import gradient, clamp, gradient_lp, YCrCb2RGB
from net import LE2Fusion
# from torch.utils.tensorboard import SummaryWriter
import torch, gc

gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
def init_seeds(seed=0):

    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LE2Fusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='/data/Disk_A/yongbiao/USERPROG/PIAFusion/datasets/msrs_train/',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='../trained-all/trained_final')  # 模型存储路径
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=30, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 50]', type=str,
                        metavar='N', help='loss weight')
    # parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth',
    #                     help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    # init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    # writer = SummaryWriter("./logs_train_fusion")
    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = LE2Fusion()
        # device = torch.device("cuda:1")
        # model.to(device)
        # model = model.to(device)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0,2]).cuda()
        # 加载预训练的分类模型


        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            # 修改学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_train = []
            list_loss_illum = []
            list_loss_aux = []
            list_loss_texture = []

            model.train()#训练
            train_tqdm = tqdm(train_loader, total=len(train_loader))

            for vis_image, vis_y_image, cb, cr, inf_image, _ in train_tqdm:

                vis_y_image = vis_y_image.cuda()
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()

################################数据处理
                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image,vis_image)#torch.Size([16, 1, 256, 256])
                # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
                fused_image = clamp(fused_image)
                # writer.add_images("input", imgs, step)

                # x_ssim = ssim_loss(fused_image, inf_image, window_size=11, reduction='none')
                y_ssim = ssim_loss(fused_image, vis_y_image, window_size=11, reduction='none')
                # s_loss = 2 - y_ssim-x_ssim
                s_loss = 1-y_ssim
                s_loss = s_loss.mean()


                ei_ir = pixel_intensity(inf_image)
                ei_vi = pixel_intensity(vis_y_image)
                ei_ir = ei_ir/9
                ei_vi = ei_vi / 9

                # loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))
                loss_aux = F.l1_loss(fused_image, torch.max(ei_ir, ei_vi))
                # loss_l1 = F.l1_loss(fused_image,ei_inf)+F.l1_loss(fused_image,ei_vis)
                loss_illum = s_loss

                ms_ssim_loss = pytorch_msssim.MS_SSIM_L1_LOSS()
                # rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                # loss_ms=ms_ssim_loss(rgb_fused_image,vis_image)

                # loss_texture
                loss_texture1 = F.l1_loss(gradient(fused_image), gradient(inf_image))
                loss_texture2 = F.l1_loss(gradient(fused_image), gradient(vis_y_image))
                loss_texture = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))



                t1, t2, t3 = eval(args.loss_weight)#[3, 7, 50]
                loss = t1*loss_illum + t2*loss_aux +49*loss_texture
                # loss = t2 * loss_aux + t3 * loss_texture
                s=loss.item()
                train_tqdm.set_postfix(epoch=epoch, ssim=loss_illum.item(),region=loss_aux.item(),
                                       texture=loss_texture.item(),
                                       total=loss.item())

                loss_train.append(loss.item())
                list_loss_illum.append(loss_illum.item())
                list_loss_aux.append(loss_aux.item())
                list_loss_texture.append(loss_texture.item())

                # writer.add_scalar("train_loss", loss.item(), epoch)
                loss.backward()
                total = sum([params.nelement() for params in model.parameters()])
                # print("Number of params: {%.2f M}" % (total / 1e6))
                optimizer.step()

            torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')





