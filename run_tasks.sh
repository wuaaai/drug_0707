#!/bin/bash
WORK_DIR=~/guanY/PersonalMed-9.15/PersonalMed/
cd "$WORK_DIR" || exit
CURRENT_DATE=$(date "+%Y-%m-%d_%H-%M-%S")
# droupt_out参数敏感性实验
# TASKS=(
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.9 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.9_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.8 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.8_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.7 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.7_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.6_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.5 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.5_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.4 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.4_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.3 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.3_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.2 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.2_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --lr 5e-4 --dim 128 --dropout 0.1 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.1_gamma0.2_step35 --gamma 0.2 --step_size 35"
# )
# TASKS=(
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.9 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.9_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.8 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.8_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.7 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.7_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.5 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.5_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.4 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.4_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.3 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.3_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.2 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.2_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --lr 5e-4 --dim 128 --dropout 0.1 --notes ${CURRENT_DATE}_lr5e-4_dim128_dropout0.1_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
# )
# batch_size参数敏感性实验
# TASKS=(
#     "python main.py --batch_size 4 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size4_lr5e-4_dim128_dropout0.6_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --batch_size 8 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size8_lr5e-4_dim128_dropout0.6_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35 --gamma 0.2 --step_size 35"
#     "python main.py --batch_size 32 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim128_dropout0.6_gamma0.2_step35 --gamma 0.2 --step_size 35"
# )
# TASKS=(
#     "python main.py --dataset mimic4 --batch_size 4 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size4_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 8 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size8_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 32 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
# )
# dim参数敏感性实验
# TASKS=(
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 32 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size4_lr5e-4_dim32_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 64 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size8_lr5e-4_dim64_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 256 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim256_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 512 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim512_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
# )
# TASKS=(
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 32 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size4_lr5e-4_dim32_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 64 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size8_lr5e-4_dim64_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 256 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim256_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 512 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size32_lr5e-4_dim512_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
# )
# lr参数敏感性实验
# TASKS=(
#     "python main.py --dataset mimic3 --batch_size 16 --lr 1e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr1e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 1e-3 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr1e-3_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-3 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-3_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
# )
# TASKS=(
#     "python main.py --dataset mimic4 --batch_size 16 --lr 1e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr1e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 1e-3 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr1e-3_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-3 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-3_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
# )
# step_size参数敏感性实验
# TASKS=(
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step15_datasetmimic3 --gamma 0.2 --step_size 15"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step20_datasetmimic3 --gamma 0.2 --step_size 20"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step25_datasetmimic3 --gamma 0.2 --step_size 25"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step30_datasetmimic3 --gamma 0.2 --step_size 30"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step31_datasetmimic3 --gamma 0.2 --step_size 31"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step32_datasetmimic3 --gamma 0.2 --step_size 32"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step33_datasetmimic3 --gamma 0.2 --step_size 33"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step34_datasetmimic3 --gamma 0.2 --step_size 34"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic3 --gamma 0.2 --step_size 35"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step36_datasetmimic3 --gamma 0.2 --step_size 36"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step37_datasetmimic3 --gamma 0.2 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step38_datasetmimic3 --gamma 0.2 --step_size 38"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step39_datasetmimic3 --gamma 0.2 --step_size 39"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step40_datasetmimic3 --gamma 0.2 --step_size 40"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step45_datasetmimic3 --gamma 0.2 --step_size 45"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step50_datasetmimic3 --gamma 0.2 --step_size 50"
# )
TASKS=(
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step15_datasetmimic4 --gamma 0.2 --step_size 15"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step20_datasetmimic4 --gamma 0.2 --step_size 20"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step25_datasetmimic4 --gamma 0.2 --step_size 25"
    "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step30_datasetmimic4 --gamma 0.2 --step_size 30"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step31_datasetmimic4 --gamma 0.2 --step_size 31"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step32_datasetmimic4 --gamma 0.2 --step_size 32"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step33_datasetmimic4 --gamma 0.2 --step_size 33"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step34_datasetmimic4 --gamma 0.2 --step_size 34"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step35_datasetmimic4 --gamma 0.2 --step_size 35"
    # "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step36_datasetmimic4 --gamma 0.2 --step_size 36"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step37_datasetmimic4 --gamma 0.2 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step38_datasetmimic4 --gamma 0.2 --step_size 38"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step39_datasetmimic4 --gamma 0.2 --step_size 39"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step40_datasetmimic4 --gamma 0.2 --step_size 40"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step45_datasetmimic4 --gamma 0.2 --step_size 45"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step50_datasetmimic4 --gamma 0.2 --step_size 50"
)
# gamma参数敏感性实验
# TASKS=(
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.1_step37_datasetmimic3 --gamma 0.1 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step37_datasetmimic3 --gamma 0.2 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.3_step37_datasetmimic3 --gamma 0.3 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.4_step37_datasetmimic3 --gamma 0.4 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.5_step37_datasetmimic3 --gamma 0.5 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.6_step37_datasetmimic3 --gamma 0.6 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.7_step37_datasetmimic3 --gamma 0.7 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.8_step37_datasetmimic3 --gamma 0.8 --step_size 37"
#     "python main.py --dataset mimic3 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.9_step37_datasetmimic3 --gamma 0.9 --step_size 37"
# )
# TASKS=(
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.1_step37_datasetmimic4 --gamma 0.1 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.2_step37_datasetmimic4 --gamma 0.2 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.3_step37_datasetmimic4 --gamma 0.3 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.4_step37_datasetmimic4 --gamma 0.4 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.5_step37_datasetmimic4 --gamma 0.5 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.6_step37_datasetmimic4 --gamma 0.6 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.7_step37_datasetmimic4 --gamma 0.7 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.8_step37_datasetmimic4 --gamma 0.8 --step_size 37"
#     "python main.py --dataset mimic4 --batch_size 16 --lr 5e-4 --dim 128 --dropout 0.6 --notes ${CURRENT_DATE}_batch_size16_lr5e-4_dim128_dropout0.6_gamma0.9_step37_datasetmimic4 --gamma 0.9 --step_size 37"
# )
# 遍历任务列表并依次执行
for TASK in "${TASKS[@]}"; do
    echo "正在执行: $TASK"
    nohup $TASK > "/dev/null" 2>&1 &
    wait # 等待当前任务完成后再执行下一个
    echo "完成: $TASK"
done

echo "所有任务已完成！"