#!/bin/bash

number_of_CPUs=16

# python3 train_DCMM.py \
#     test=False \
#     task=Tracking \
#     num_envs=${number_of_CPUs}

path_to_tracking_model="./assets/models/track1.pth"
python3 train_DCMM.py \
    viewer=False \
    test=False \
    task=Catching_TwoStage \
    num_envs=${number_of_CPUs} \
    checkpoint_tracking=${path_to_tracking_model}