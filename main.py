import tensorflow as tf
import numpy as np
import pdb


from model import REG
from os.path import join
from configuration import get_config
import wandb


def main(config=None):
    with wandb.init(entity="musktang", config=config) as run:
        selfconfig = get_config()

        log_path = selfconfig.log_path
        saver_dir = selfconfig.saver_path

        # ===========================================================
        # ===========             Main Model             ============
        # ===========================================================
        print('--- Build Model ---')
        note = selfconfig.note
        date = selfconfig.date
        gpu_index = selfconfig.gpu_index
        learning_rate = selfconfig.init_learning_rate
        read_ckpt = selfconfig.read_ckpt
        model = REG(log_path, saver_dir, date, gpu_index, note, selfconfig, config)

        print('init learning rate = ' + str(learning_rate))
        model.build(reuse=False)
        # model.buildModule2(reuse=False)

        print('--- Train Model ---')                           #'saver_DDAE/0318'
        model.train(read_ckpt=read_ckpt)

def model_sweep():
    sweep_config={"method":"random"}
    param_dict = {
        "batch_size":{
            "values":[64, 128, 256]
        },
        "filter_h":{
            "values":[3, 5, 7]
        },
        "filter_w": {
            "values": [1, 2, 3]
        },
        "l1_output_num": {
            "values": [20, 40, 60]
        },
        "l2_output_num": {
            "values": [10, 30, 50]
        },
        "l3_output_num": {
            "values": [10, 30, 50]
        },
        "activate": {
            "values": ["leaky_relu", "relu", "tanh", "selu"]
        },
        "n_mels": {
            "values": [24, 48, 72]
        },
    }
    sweep_config["parameters"] = param_dict

    sweep_id = wandb.sweep(sweep_config, project="VAD_sweep_2")
    wandb.agent(sweep_id, main, count=30)

if __name__ == '__main__':
    # main()
    model_sweep()