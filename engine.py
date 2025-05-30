import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

def cam_max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C, 1, 1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C, 1, 1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N, C, 1, 1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N, C, 1, 1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        return p
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p [p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p [p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        return p

<<<<<<< HEAD
=======
def correspondence_loss(img_patch, img_patch2):
    image_patch = F.normalize(img_patch,dim=1)
    image_patch2 = F.normalize(img_patch2.detach(),dim=1)
    loss_scale = image_patch*image_patch2
    loss_scale = torch.sum(loss_scale, dim=1)
    loss_scale = torch.sigmoid(loss_scale)
    loss_scale = -torch.log(loss_scale+1e-5)
    loss_scale = torch.mean(loss_scale)
    return loss_scale
>>>>>>> d02aef2 (update)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, samples_aug, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        samples_aug = samples_aug.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        #samples2 = F.interpolate(samples, scale_factor=0.5, mode='bilinear', align_corners=True)
        patch_outputs = None
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs2 = model(samples_aug, strong_aug=True)
<<<<<<< HEAD
            if not isinstance(outputs, torch.Tensor):
                outputs, patch_outputs, mtatt = outputs
                outputs2, patch_outputs2, mtatt2 = outputs2
=======
            #outputs2 = model(samples_aug, strong_aug=True)
            if not isinstance(outputs, torch.Tensor):
                outputs, patch_outputs, mtatt, img_patch = outputs
                outputs2, patch_outputs2, mtatt2, img_patch2 = outputs2
>>>>>>> d02aef2 (update)
            
            loss = F.multilabel_soft_margin_loss(outputs, targets)
            loss2 = F.multilabel_soft_margin_loss(outputs2, targets)
            #mtatt = F.interpolate(cam_max_norm(mtatt),scale_factor=0.5,mode='bilinear',align_corners=True)*targets.unsqueeze(2).unsqueeze(3)
<<<<<<< HEAD
            mtatt2 = cam_max_norm(mtatt2)*targets.unsqueeze(2).unsqueeze(3)
            loss_scale = torch.mean(torch.abs(mtatt-mtatt2))
            metric_logger.update(cls_loss=loss.item())
            metric_logger.update(cls2_loss=loss2.item())
            metric_logger.update(scale_loss=loss_scale.item())
            loss = loss + loss2
=======
            img_patch = cam_max_norm(img_patch)*targets.unsqueeze(2).unsqueeze(3)
            mtatt = cam_max_norm(mtatt)*targets.unsqueeze(2).unsqueeze(3)
            img_patch2 = cam_max_norm(img_patch2)*targets.unsqueeze(2).unsqueeze(3)
            mtatt2 = cam_max_norm(mtatt2)*targets.unsqueeze(2).unsqueeze(3)
            # loss_scale1 = correspondence_loss(img_patch, img_patch2)
            # loss_scale2 = correspondence_loss(img_patch2, img_patch)
            loss_scale1 = torch.mean(torch.abs(mtatt- mtatt2.detach()))
            loss_scale2 = torch.mean(torch.abs(img_patch- img_patch2.detach()))
            #loss_scale2 = torch.mean(torch.abs(img_patch.detach() - img_patch2))
            
            metric_logger.update(cls_loss=loss.item())
            metric_logger.update(cls2_loss=loss2.item())
            metric_logger.update(scale_loss1=loss_scale1.item())
            metric_logger.update(scale_loss2=loss_scale2.item())
            loss = loss + loss2 + 0.5*(0.5*loss_scale1 + 0.5*loss_scale2)
            #loss = loss + loss2 + loss_scale1 + loss_scale2
>>>>>>> d02aef2 (update)
            if  patch_outputs is not None:

                ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                ploss2 = F.multilabel_soft_margin_loss(patch_outputs2, targets)
                metric_logger.update(pat2_loss=ploss2.item())
                loss = loss + ploss+ploss2

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            if not isinstance(output, torch.Tensor):
<<<<<<< HEAD
                output, patch_output, mtatt = output
=======
                output, patch_output, mtatt, img_patch = output
>>>>>>> d02aef2 (update)
            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)


        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mAP=metric_logger.mAP, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(os.path.join(args.img_list, 'train_aug_id.txt')).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # for iter, (image_list, target) in enumerate(data_loader):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size


        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                elif 'MCTformerV2' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type, strong_aug=args.strong_aug)
                    patch_attn = torch.sum(patch_attn, dim=0)


                if args.patch_attn_refine:
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order='C')
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ('_%s' % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            np.save(os.path.join(folder, img_name + '.npy'), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)