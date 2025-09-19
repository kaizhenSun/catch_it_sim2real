#!/bin/bash

path_to_tracking_model="assets/models/track1.pth"
path_to_catching_model="assets/models/catch_two_stage1.pth"
# python3 train_DCMM.py \
#     test=True \
#     task=Tracking \
#     num_envs=1 \
#     checkpoint_tracking=${path_to_tracking_model} \
#     object_eval=True \
#     viewer=True \
#     imshow_cam=False

# python3 train_DCMM.py \
#     test=True \
#     task=Catching_TwoStage \
#     num_envs=1 \
#     checkpoint_catching=${path_to_catching_model} \
#     object_eval=True \
#     viewer=True \
#     imshow_cam=False

python3 train_DCMM_sim2real.py \
    test=True \
    task=Catching_TwoStage \
    num_envs=1 \
    checkpoint_catching=${path_to_catching_model} \
    object_eval=True \
    viewer=True \
    imshow_cam=False