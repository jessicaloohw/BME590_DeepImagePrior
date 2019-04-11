#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python training.py '1' 'deep_decoder' 'mse' 'adam' 0.0001 10 1
#python training.py '2_R'
#python training.py '3'
#python training.py '4'
#python training.py '5'
#python training.py '6'
#python training.py '7'
#python training.py '8'
#python training.py '9'
#python training.py '10'
#python training.py '11'
#python training.py '12'
#python training.py '13'
#python training.py '14'
#python training.py '15'
#python training.py '16'
#python training.py '17'
#python training.py 'D_1'
#python training.py 'D_2'
#python training.py 'D_3'
#python training.py 'V22'
#python training.py 'V_1'