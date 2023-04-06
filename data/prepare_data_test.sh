#!/usr/bin/env bash

# prepare data to test the SR model from 16 to 128
python prepare_data.py --path J:/Mississippi_design_storm/Super_resolution/SR_diffusion/datasets/own_dataset_div2k_hr_0403 --out J:/Mississippi_design_storm/Super_resolution/SR_diffusion/datasets/div2k_16_128_0403 --size 16,128