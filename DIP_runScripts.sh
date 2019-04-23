#!/bin/bash

python training.py '1' 'unet' 'mse' 'adam' 0.0001 50000 1000
python training.py '2_R' 'unet' 'mse' 'adam' 0.0001 50000 1000
python training.py '3' 'unet' 'mse' 'adam' 0.0001 50000 1000
python training.py '4' 'unet' 'mse' 'adam' 0.0001 50000 1000
python training.py '5' 'unet' 'mse' 'adam' 0.0001 50000 1000

python training.py '1' 'unet' 'l1' 'adam' 0.0001 50000 1000
python training.py '2_R' 'unet' 'l1' 'adam' 0.0001 50000 1000
python training.py '3' 'unet' 'l1' 'adam' 0.0001 50000 1000
python training.py '4' 'unet' 'l1' 'adam' 0.0001 50000 1000
python training.py '5' 'unet' 'l1' 'adam' 0.0001 50000 1000

python training.py '1' 'unet' 'mse_l1' 'adam' 0.0001 50000 1000
python training.py '2_R' 'unet' 'mse_l1' 'adam' 0.0001 50000 1000
python training.py '3' 'unet' 'mse_l1' 'adam' 0.0001 50000 1000
python training.py '4' 'unet' 'mse_l1' 'adam' 0.0001 50000 1000
python training.py '5' 'unet' 'mse_l1' 'adam' 0.0001 50000 1000

python training.py '1' 'unet' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '2_R' 'unet' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '3' 'unet' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '4' 'unet' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '5' 'unet' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0

python training.py '1' 'unet' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '2_R' 'unet' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '3' 'unet' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '4' 'unet' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '5' 'unet' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0

python training.py '1' 'deep_decoder' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '2_R' 'deep_decoder' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '3' 'deep_decoder' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '4' 'deep_decoder' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0
python training.py '5' 'deep_decoder' 'mse_with_tv_reg' 'adam' 0.0001 50000 1000 1e-6 0.0 10.0

python training.py '1' 'deep_decoder' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '2_R' 'deep_decoder' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '3' 'deep_decoder' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '4' 'deep_decoder' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
python training.py '5' 'deep_decoder' 'mse_with_edge_reg' 'adam' 0.0001 50000 1000 1e-6 1e-6 10.0
