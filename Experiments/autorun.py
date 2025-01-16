import sys
import os

sys.path.append('F:\Bytedance\ByteDance_CTR')
os.chdir('F:\Bytedance\ByteDance_CTR')
os.environ['PYTHONPATH'] = "F:\Bytedance\ByteDance_CTR"

commands_LR = [
    "python Models/LR.py --model_name LR --lr 0.001 --reg 1e-4 --treatment --cuda",
    "python Models/LR.py --model_name LR --lr 0.01 --reg 1e-4 --treatment --cuda",
    "python Models/LR.py --model_name LR --lr 0.1 --reg 1e-4 --treatment --cuda",

    "python Models/LR.py --model_name LR --lr 0.01 --reg 5e-4 --treatment --cuda",
    "python Models/LR.py --model_name LR --lr 0.01 --reg 1e-3 --treatment --cuda",
    "python Models/LR.py --model_name LR --lr 0.01 --reg 1e-2 --treatment --cuda"
]

commands_FM = [
    "python Models/FM.py --model_name FM --lr 0.001 --reg 1e-4 --k 10 --treatment --cuda",
    "python Models/FM.py --model_name FM --lr 0.01 --reg 1e-4 --k 10 --treatment --cuda",
    "python Models/FM.py --model_name FM --lr 0.10 --reg 1e-4 --k 10 --treatment --cuda",

    "python Models/FM.py --model_name FM --lr 0.01 --reg 1e-4 --k 13 --treatment --cuda",
    "python Models/FM.py --model_name FM --lr 0.01 --reg 5e-4 --k 10 --treatment --cuda",
    "python Models/FM.py --model_name FM --lr 0.01 --reg 1e-3 --k 10 --treatment --cuda"

]

commands_FFM = [
    "python Models/FFM.py --model_name FFM --lr 0.001 --reg 1e-4 --k 10 --treatment --cuda",
    "python Models/FFM.py --model_name FFM --lr 0.01 --reg 1e-4 --k 10 --treatment --cuda",
    "python Models/FFM.py --model_name FFM --lr 0.10 --reg 1e-4 --k 10 --treatment --cuda",

    "python Models/FFM.py --model_name FFM --lr 0.01 --reg 1e-4 --k 13 --treatment --cuda",
    "python Models/FFM.py --model_name FFM --lr 0.01 --reg 5e-4 --k 10 --treatment --cuda",
    "python Models/FFM.py --model_name FFM --lr 0.01 --reg 1e-3 --k 10 --treatment --cuda"

]

commands_WD= [
    "python Models/WD.py --model_name WD --lr 0.001 --reg 1e-4 --treatment --cuda",
    "python Models/WD.py --model_name WD --lr 0.001 --reg 5e-4 --treatment --cuda",
    "python Models/WD.py --model_name WD --lr 0.001 --reg 1e-3 --treatment --cuda",

    "python Models/WD.py --model_name WD --lr 0.01 --reg 1e-4 --treatment --cuda",
    "python Models/WD.py --model_name WD --lr 0.01 --reg 5e-4 --treatment --cuda",
    "python Models/WD.py --model_name WD --lr 0.01 --reg 1e-3 --treatment --cuda"

]

commands_DFM = [
    "python Models/DFM.py --model_name DFM --lr 0.001 --reg 1e-4 --emb_dim 32 --treatment --cuda",
    "python Models/DFM.py --model_name DFM --lr 0.001 --reg 5e-4 --emb_dim 32 --treatment --cuda",
    "python Models/DFM.py --model_name DFM --lr 0.001 --reg 1e-3 --emb_dim 32 --treatment --cuda",

    "python Models/DFM.py --model_name DFM --lr 0.01 --reg 1e-4 --emb_dim 32 --treatment --cuda",
    "python Models/DFM.py --model_name DFM --lr 0.01 --reg 5e-4 --emb_dim 32 --treatment --cuda",
    "python Models/DFM.py --model_name DFM --lr 0.01 --reg 1e-3 --emb_dim 32 --treatment --cuda",

    "python Models/DFM.py --model_name DFM --lr 0.01 --reg 1e-4 --emb_dim 16 --treatment --cuda",
    "python Models/DFM.py --model_name DFM --lr 0.01 --reg 1e-4 --emb_dim 32 --treatment --cuda",
    "python Models/DFM.py --model_name DFM -ddd-lr 0.01 --reg 1e-4 --emb_dim 64 --treatment --cuda"
]

commands_DCN= [
    "python Models/DCN.py --model_name DCN --lr 0.01 --reg 1e-4 --cross 16 --treatment --cuda",
    "python Models/DCN.py --model_name DCN --lr 0.01 --reg 5e-4 --cross 16 --treatment --cuda",
    "python Models/DCN.py --model_name DCN --lr 0.01 --reg 1e-3 --cross 16 --treatment --cuda",

    "python Models/DCN.py --model_name DCN --lr 0.001 --reg 1e-4 --cross 16 --treatment --cuda",
    # "python Models/DCN.py --model_name DCN --lr 0.1 --reg 1e-4 --cross 16 --treatment --cuda",# This wont work due to large lr

    "python Models/DCN.py --model_name DCN --lr 0.01 --reg 1e-4 --cross 8 --treatment --cuda",
    # "python Models/DCN.py --model_name DCN --lr 0.01 --reg 1e-4 --cross 10 --treatment --cuda"

]

commands_DIN= [
    "python Models/DIN.py --model_name DIN --lr 0.001 --reg 1e-4 --treatment --cuda",
    "python Models/DIN.py --model_name DIN --lr 0.01 --reg 1e-4 --treatment --cuda",
    "python Models/DIN.py --model_name DIN --lr 0.1 --reg 1e-4 --treatment --cuda",

    "python Models/DIN.py --model_name DIN --lr 0.01 --reg 1e-4 --treatment --cuda",
    "python Models/DIN.py --model_name DIN --lr 0.01 --reg 5e-4 --treatment --cuda",
    "python Models/DIN.py --model_name DIN --lr 0.01 --reg 1e-3 --treatment --cuda"

]

# print(os.getcwd())  # Check the current working directory
# print(os.listdir())  # Check what is in this directory

for cmd in commands_DCN:
    print(f"Running: {cmd}")
    os.system(cmd)

# run tensorboard
# tensorboard --logdir=./Results/xx/xx_event