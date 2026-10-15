import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add paths
sys.path.insert(0, '/workspace/python_pro/test1/RopStitch/wCoefNet/Codes')
sys.path.insert(0, '/workspace/python_pro/test1/RopStitch/M4')

from network import Network, CoefNetwork, get_rigid_mesh, get_norm_mesh, H2Mesh
from modules_b import CrossAttentionMatcher, ConvMeshHead
from loss_m3 import M3StitchingLoss
from loss import mesh_folding_loss, mesh_bending_loss
from dataset import TrainDataset
import grid_res
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

MODEL_DIR = '/workspace/python_pro/test1/RopStitch/M4/model_b'
SUMMARY_DIR = '/workspace/python_pro/test1/RopStitch/M4/summary_b'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=SUMMARY_DIR)


def forward_combo_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, input1, input2, alpha=0.5):
    """
    Combo B Forward:
    1. Extract multi-scale deep features from dual ResNet backbones
    2. Homography alignment & CoefNet decomposition
    3. Bilateral Cross-Attention matching enhancement
    4. Fully Convolutional Mesh Regression (ConvMeshHead) -> predicts [B, 13, 13, 2]
    5. TPS Warping & Alignment
    """
    B, _, img_h, img_w = input1.size()
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]], device=input1.device).unsqueeze(0).expand(B, -1, -1)

    # 1. Feature extraction
    with torch.no_grad():
        _, _, frozen_feat_2_64, _ = net.fronzen_resnet_backbone(input1, input2)
    active_feat_1_64, active_feat_1_32, active_feat_2_64, active_feat_2_32 = net.active_resnet_backbone(input1, input2)

    c_32 = (1 - alpha) * net.CCL(net.fronzen_resnet_backbone(input1, input2)[1], net.fronzen_resnet_backbone(input1, input2)[3]) \
           + alpha * net.CCL(active_feat_1_32, active_feat_2_32)
    t_1 = net.regressHomo_part1(c_32).view(B, -1)
    offset_1 = net.regressHomo_part2(t_1)
    H_motion = offset_1.reshape(-1, 4, 2)

    # 2. CoefNet decomposition
    dst_p_ref_list, dst_p_tgt_list, final_coef_list = coef_net(input1, input2, H_motion)
    c0 = final_coef_list[-1]
    rigid_mesh = get_rigid_mesh(B, img_h, img_w)

    dst_p_ref = dst_p_ref_list[-1]
    dst_p_tgt = dst_p_tgt_list[-1]

    H_ref = torch_DLT.Tensor_ACA_rect(src_p, dst_p_ref)
    H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)

    # Feature warping by updated homography
    H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_ref/8)
    H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)

    M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                             [0., img_h/8 / 2.0, img_h/8 / 2.0],
                             [0., 0., 1.]], device=input1.device).unsqueeze(0).expand(B, -1, -1)
    M_inv = torch.inverse(M_tensor)

    H_mat_ref = torch.matmul(torch.matmul(M_inv, H_ref_feat), M_tensor)
    warp_f1_ref = torch_homo_transform.transformer(active_feat_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
    H_mat_tgt = torch.matmul(torch.matmul(M_inv, H_tgt_feat), M_tensor)
    warp_f2_tgt = torch_homo_transform.transformer(active_feat_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

    # 3. Bilateral Cross-Attention Enhancement
    warp_f1_enh, warp_f2_enh = cross_matcher(warp_f1_ref, warp_f2_tgt)

    # 4. Fully Convolutional Mesh Regression (ConvMeshHead - No Flatten, No Dense Layers)
    motion_ref = conv_head_ref(warp_f1_enh)  # [B, 13, 13, 2]
    motion_tgt = conv_head_tgt(warp_f2_enh)  # [B, 13, 13, 2]

    mesh_ref = ini_mesh_ref + motion_ref
    mesh_tgt = ini_mesh_tgt + motion_tgt

    # 5. TPS Image Warping
    norm_rigid = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    norm_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)
    mask = torch.ones_like(input2)

    output_ref = torch_tps_transform.transformer(torch.cat((input1+1, mask), 1), norm_ref, norm_rigid, (img_h, img_w))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2+1, mask), 1), norm_tgt, norm_rigid, (img_h, img_w))

    return {
        'rigid_mesh': rigid_mesh,
        'mesh_ref': mesh_ref,
        'mesh_tgt': mesh_tgt,
        'motion_ref': motion_ref,
        'motion_tgt': motion_tgt,
        'output_ref': output_ref,
        'output_tgt': output_tgt,
    }


def train_combo_b(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"Loading dataset from {args.train_path}...")
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    net = Network().cuda()
    coef_net = CoefNetwork().cuda()
    cross_matcher = CrossAttentionMatcher(in_channels=128).cuda()
    conv_head_ref = ConvMeshHead(in_channels=128).cuda()
    conv_head_tgt = ConvMeshHead(in_channels=128).cuda()

    # Load baseline weights
    homo_checkpoint = torch.load(args.woCoefNet_path, map_location='cuda')
    net.load_state_dict(homo_checkpoint['model'])

    if args.resume_coef_path and os.path.exists(args.resume_coef_path):
        c_ckpt = torch.load(args.resume_coef_path, map_location='cuda')
        coef_net.load_state_dict(c_ckpt['model'])
        print(f"Loaded pretrained coef_net from {args.resume_coef_path}")

    # Unfreezing strategy: Active backbone unfreezed with small lr, new modules with full lr
    for p in net.fronzen_resnet_backbone.parameters():
        p.requires_grad = False
    for p in net.active_resnet_backbone.parameters():
        p.requires_grad = True

    criterion = M3StitchingLoss(w_l1=args.w_l1, w_ssim=args.w_ssim, w_smooth=0.05).cuda()

    # Layer-wise optimizer
    optimizer = optim.Adam([
        {'params': cross_matcher.parameters(), 'lr': args.lr},
        {'params': conv_head_ref.parameters(), 'lr': args.lr},
        {'params': conv_head_tgt.parameters(), 'lr': args.lr},
        {'params': coef_net.parameters(), 'lr': args.lr * 0.2},
        {'params': net.active_resnet_backbone.parameters(), 'lr': args.lr * 0.05},
        {'params': net.regressHomo_part1.parameters(), 'lr': args.lr * 0.05},
        {'params': net.regressHomo_part2.parameters(), 'lr': args.lr * 0.05},
    ], betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=args.lr * 0.05)

    print(f"================== Start Combo B Training (Cross-Attention + ConvMeshHead) on GPU {args.gpu} ==================")
    score_print_freq = 50
    glob_iter = 0

    for epoch in range(args.max_epoch):
        net.train()
        coef_net.train()
        cross_matcher.train()
        conv_head_ref.train()
        conv_head_tgt.train()

        loss_total_acc = 0.0
        loss_l1_acc = 0.0
        loss_ssim_acc = 0.0
        loss_geom_acc = 0.0

        for i, batch_value in enumerate(train_loader):
            input1 = batch_value[0].float().cuda()
            input2 = batch_value[1].float().cuda()

            optimizer.zero_grad()
            res = forward_combo_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, input1, input2)

            out_ref = res['output_ref']
            out_tgt = res['output_tgt']
            mesh_ref = res['mesh_ref']
            mesh_tgt = res['mesh_tgt']
            rigid_mesh = res['rigid_mesh']
            motion_ref = res['motion_ref']
            motion_tgt = res['motion_tgt']

            m3_loss, loss_details = criterion(out_ref, out_tgt, motion_ref, motion_tgt)

            # Balanced topological regularization (anti-folding)
            loss_fold = mesh_folding_loss(mesh_ref, rigid_mesh) + mesh_folding_loss(mesh_tgt, rigid_mesh)
            loss_bend = mesh_bending_loss(mesh_ref, rigid_mesh) + mesh_bending_loss(mesh_tgt, rigid_mesh)
            geom_loss = 0.5 * loss_fold + 0.5 * loss_bend

            total_loss = m3_loss + geom_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cross_matcher.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(conv_head_ref.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(conv_head_tgt.parameters(), max_norm=2.0)
            optimizer.step()

            loss_total_acc += total_loss.item()
            loss_l1_acc += loss_details['loss_l1']
            loss_ssim_acc += loss_details['loss_ssim']
            loss_geom_acc += geom_loss.item()

            if (i + 1) % score_print_freq == 0:
                avg_tot = loss_total_acc / score_print_freq
                avg_l1 = loss_l1_acc / score_print_freq
                avg_ssim = loss_ssim_acc / score_print_freq
                avg_geom = loss_geom_acc / score_print_freq

                print(f"Combo-B [Ep {epoch+1:02d}/{args.max_epoch:02d}][It {i+1:04d}/{len(train_loader):04d}] "
                      f"Tot: {avg_tot:.4f} | L1: {avg_l1:.4f} | SSIM_loss: {avg_ssim:.4f} | Geom: {avg_geom:.4f}", flush=True)

                writer.add_scalar('ComboB/Total_Loss', avg_tot, glob_iter)
                writer.add_scalar('ComboB/L1_Loss', avg_l1, glob_iter)
                writer.add_scalar('ComboB/SSIM_Loss', avg_ssim, glob_iter)
                writer.add_scalar('ComboB/Geom_Loss', avg_geom, glob_iter)

                loss_total_acc, loss_l1_acc, loss_ssim_acc, loss_geom_acc = 0, 0, 0, 0

            glob_iter += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            save_path = os.path.join(MODEL_DIR, f"epoch{epoch+1:03d}_combob_model.pth")
            torch.save({
                'cross_matcher': cross_matcher.state_dict(),
                'conv_head_ref': conv_head_ref.state_dict(),
                'conv_head_tgt': conv_head_tgt.state_dict(),
                'coef_net': coef_net.state_dict(),
                'active_backbone': net.active_resnet_backbone.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, save_path)
            print(f"===> Saved Combo B checkpoint: {save_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.8)
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--resume_coef_path', type=str, default='/workspace/python_pro/test1/RopStitch/wCoefNet/model_coef/epoch050_coefmodel.pth')
    parser.add_argument('--train_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/training/')

    args = parser.parse_args()
    train_combo_b(args)
