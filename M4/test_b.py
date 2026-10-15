import argparse
import os
import sys
import cv2
import numpy as np
import time
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim

# Add paths
sys.path.insert(0, '/workspace/python_pro/test1/RopStitch/wCoefNet/Codes')
sys.path.insert(0, '/workspace/python_pro/test1/RopStitch/M4')

from network import Network, CoefNetwork, H2Mesh, get_rigid_mesh, get_norm_mesh, resize_512
from modules_b import CrossAttentionMatcher, ConvMeshHead
from dataset import TestDataset
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def maskSSIM(image1, image2, mask):
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)
    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim


def maskPSNR(image1, image2, mask):
    image1 = image1 * mask / 255.
    image2 = image2 * mask / 255.
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / (mask.sum() + 1e-6))
    if rmse == 0:
        return 100.0
    psnr = 20 * np.log10(1.0 / rmse)
    return psnr


def build_output_combo_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1_tensor, in2_tensor, alpha=0.5, max_out_height=4000):
    batch_size, _, img_h, img_w = in1_tensor.size()

    resized_input1 = resize_512(in1_tensor)
    resized_input2 = resize_512(in2_tensor)

    with torch.no_grad():
        _, _, frozen_feat_2_64, _ = net.fronzen_resnet_backbone(resized_input1, resized_input2)
        active_feat_1_64, active_feat_1_32, active_feat_2_64, active_feat_2_32 = net.active_resnet_backbone(resized_input1, resized_input2)

        c_32 = (1 - alpha) * net.CCL(net.fronzen_resnet_backbone(resized_input1, resized_input2)[1], net.fronzen_resnet_backbone(resized_input1, resized_input2)[3]) \
               + alpha * net.CCL(active_feat_1_32, active_feat_2_32)
        t_1 = net.regressHomo_part1(c_32).view(batch_size, -1)
        offset_1 = net.regressHomo_part2(t_1)
        H_motion = offset_1.reshape(-1, 4, 2)

        dst_p_ref_list, dst_p_tgt_list, final_coef_list = coef_net(resized_input1, resized_input2, H_motion)
        c0 = final_coef_list[-1]
        rigid_mesh_512 = get_rigid_mesh(batch_size, 512, 512)

        src_p = torch.tensor([[0., 0.], [512., 0.], [0., 512.], [512., 512.]], device=in1_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
        dst_p_ref = dst_p_ref_list[-1]
        dst_p_tgt = dst_p_tgt_list[-1]

        H_ref = torch_DLT.Tensor_ACA_rect(src_p, dst_p_ref)
        H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
        ini_mesh_ref = H2Mesh(H_ref, rigid_mesh_512)
        ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh_512)

        H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_ref/8)
        H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)

        M_tensor = torch.tensor([[512/8 / 2.0, 0., 512/8 / 2.0],
                                 [0., 512/8 / 2.0, 512/8 / 2.0],
                                 [0., 0., 1.]], device=in1_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
        M_inv = torch.inverse(M_tensor)

        H_mat_ref = torch.matmul(torch.matmul(M_inv, H_ref_feat), M_tensor)
        warp_f1_ref = torch_homo_transform.transformer(active_feat_1_64, H_mat_ref, (int(512/8), int(512/8)))
        H_mat_tgt = torch.matmul(torch.matmul(M_inv, H_tgt_feat), M_tensor)
        warp_f2_tgt = torch_homo_transform.transformer(active_feat_2_64, H_mat_tgt, (int(512/8), int(512/8)))

        # Cross-Attention enhancement
        warp_f1_enh, warp_f2_enh = cross_matcher(warp_f1_ref, warp_f2_tgt)

        # ConvMeshHead prediction
        motion_ref1 = conv_head_ref(warp_f1_enh)
        motion_tgt1 = conv_head_tgt(warp_f2_enh)

        mesh_ref1 = ini_mesh_ref + motion_ref1
        mesh_tgt1 = ini_mesh_tgt + motion_tgt1

        # Scale mesh to target resolution
        mesh_ref = torch.stack([mesh_ref1[..., 0] * img_w / 512.0, mesh_ref1[..., 1] * img_h / 512.0], 3)
        mesh_tgt = torch.stack([mesh_tgt1[..., 0] * img_w / 512.0, mesh_tgt1[..., 1] * img_h / 512.0], 3)
        rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)

        width_max = torch.maximum(torch.max(mesh_ref[..., 0]), torch.max(mesh_tgt[..., 0]))
        width_min = torch.minimum(torch.min(mesh_ref[..., 0]), torch.min(mesh_tgt[..., 0]))
        height_max = torch.maximum(torch.max(mesh_ref[..., 1]), torch.max(mesh_tgt[..., 1]))
        height_min = torch.minimum(torch.min(mesh_ref[..., 1]), torch.min(mesh_tgt[..., 1]))

        out_width = width_max - width_min
        out_height = height_max - height_min

        if max(out_height, out_width) >= max_out_height:
            return None, False

        mesh_trans_ref = torch.stack([mesh_ref[..., 0] - width_min, mesh_ref[..., 1] - height_min], 3)
        mesh_trans_tgt = torch.stack([mesh_tgt[..., 0] - width_min, mesh_tgt[..., 1] - height_min], 3)

        norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
        norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
        norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

        mask = torch.ones_like(in2_tensor)
        output_ref = torch_tps_transform.transformer(torch.cat((in1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
        output_tgt = torch_tps_transform.transformer(torch.cat((in2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    out_dict = {
        'output_ref': output_ref,
        'output_tgt': output_tgt,
        'mesh_ref': mesh_ref,
        'mesh_tgt': mesh_tgt,
        'rigid_mesh': rigid_mesh,
        'out_height': int(out_height.item() if isinstance(out_height, torch.Tensor) else out_height),
        'out_width': int(out_width.item() if isinstance(out_width, torch.Tensor) else out_width),
    }
    return out_dict, True


def test_once_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1, in2, alpha=0.5, max_out_height=4000):
    with torch.no_grad():
        batch_out, ok = build_output_combo_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1, in2, alpha=alpha, max_out_height=max_out_height)
    if not ok:
        mask = torch.ones_like(in1).to(in1.device)
        output_tps_ref = torch.cat((in1+1, mask), 1)
        output_tps_tgt = torch.cat((in2+1, mask), 1)
        warp_fallback = True
    else:
        output_tps_ref = batch_out['output_ref']
        output_tps_tgt = batch_out['output_tgt']
        warp_fallback = False

    output_ref = (output_tps_ref[0, 0:3, ...] * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    output_tgt = (output_tps_tgt[0, 0:3, ...] * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    overlap_mask = output_tps_ref[0, 3:6, ...] * output_tps_tgt[0, 3:6, ...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1, 2, 0)

    ssim = maskSSIM(output_ref, output_tgt, overlap_mask)
    psnr = maskPSNR(output_ref, output_tgt, overlap_mask)
    fusion = output_ref * (output_ref / (output_ref + output_tgt + 1e-6)) + output_tgt * (output_tgt / (output_ref + output_tgt + 1e-6))

    return ssim, psnr, fusion, {'warp_fallback': warp_fallback}


def ternary_search_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1, in2, low=-1.0, high=2.0, max_iter=5, max_out_height=4000, tol=1e-6):
    best_ssim = -float('inf')
    best_psnr = -float('inf')
    best_img = None
    best_alpha = 0.5

    for iter_idx in range(max_iter):
        if high - low < tol:
            break
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0

        ssim1, psnr1, img1, _ = test_once_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1, in2, alpha=mid1, max_out_height=max_out_height)
        ssim2, psnr2, img2, _ = test_once_b(net, coef_net, cross_matcher, conv_head_ref, conv_head_tgt, in1, in2, alpha=mid2, max_out_height=max_out_height)

        if ssim1 > best_ssim:
            best_ssim = ssim1
            best_psnr = psnr1
            best_alpha = mid1
            best_img = img1
        if ssim2 > best_ssim:
            best_ssim = ssim2
            best_psnr = psnr2
            best_alpha = mid2
            best_img = img2

        if ssim1 < ssim2:
            low = mid1
        else:
            high = mid2

    return best_alpha, best_ssim, best_psnr, best_img
