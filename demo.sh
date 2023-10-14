#!/bin/bash

iou_thresh=0.25
recall_thresh=0.5
feature_similarity_thresh=0.75

python src/proposed_fusion.py \
    --dataset "demo_dataset" \
    --video "tabletop" \
    --detic_exp "imagenet21k-0.3" \
    --iou_thresh "${iou_thresh}" \
    --recall_thresh "${recall_thresh}" \
    --feature_similarity_thresh "${feature_similarity_thresh}" \
    --depth_thresh 0.04 \
    --interval 300 \
    --visibility_thresh 0.2 \
    --size_thresh 50 \
    --device "cuda:0" \
    --output_file "proposed_fusion_detic.pkl" \
    --vocab_feature_file "src/scannet200.npy" \
    --no-sam \
    --tqdm \
    --vis \


python src/instance_query.py \
    --dataset "demo_dataset" \
    --video "tabletop" \
    --detic_exp "imagenet21k-0.3" \
    --prediction_file "proposed_fusion_detic.pkl"
