#!/usr/bin/env bash
#!/bin/bash




CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_MCTformerV2_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list  /data1/VOCdevkit/VOC2012/ImageSets/Segmentation \
                --data-path  /data1/VOCdevkit/VOC2012 \
                --output_dir AETN_results/AETN_outuput \
                --resume /data1/AETN/AETN_results/checkpoint_0.65033.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --attention-dir AETN_results/AETN_outuput/attn-patchrefine \
                --cam-npy-dir AETN_results/AETN_outuput/attn-patchrefine-npy \
# #
######### Evaluating the generated class-specific localization maps ##########
CUDA_VISIBLE_DEVICES=0 python evaluation.py --list /data1/VOCdevkit/VOC2012/ImageSets/Segmentation/train_id.txt \
                    --gt_dir /data1/VOCdevkit/VOC2012/SegmentationClassAug \
                    --logfile AETN_results/AETN_outuput/attn-patchrefine-npy/evallog.txt \
                    --type npy \
                    --curve True \
                    --predict_dir AETN_results/AETN_outuput/attn-patchrefine-npy \
                    --comment "train1464"

