import tensorflow as tf
import numpy as np
import pdb


from model import REG
from os.path import join
from configuration import get_config

#np.random.seed(1234567)

def main():
    config = get_config()

    log_path = config.log_path
    saver_dir = config.saver_path
    epochs = config.epochs
    batch_size = config.batch_size

    '''
    print('--- Synthesize Testing Noisy Data ---')
    test_noisy_dir = join(noisy_dir, 'test')
    sr_clean = 16000
    sr_noise = 16000
    data_num = None # set data_num to make testing data numbers for different snr
    snr_list = ['10dB']
    syn_test = Synth(clean_test_list, noise_test_list, sr_clean, sr_noise)
    syn_test.gen_noisy(snr_list, test_noisy_dir,
                        data_num=data_num, ADD_CLEAN=True, cpu_cores=ncores)
    '''

    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    print('--- Build Model ---')
    note = config.note
    date = config.date
    gpu_index = config.gpu_index
    learning_rate = config.init_learning_rate
    read_ckpt = config.read_ckpt

    model = REG(log_path, saver_dir, date, gpu_index, note, config)

    print('init learning rate = ' + str(learning_rate))
    model.build(reuse=False)
    # model.buildModule2(reuse=False)

    print('--- Train Model ---')                           #'saver_DDAE/0318'
    model.train(read_ckpt=read_ckpt)


if __name__ == '__main__':
    main()
