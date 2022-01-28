import tensorflow as tf
import h5py
import numpy as np
import scipy
from multiprocessing import Pool
from utils import _gen_VAD_training_data_runtime
from glob import iglob
from functools import partial
import csv
import time
import os
from os.path import join
from tqdm import tqdm
import xlrd, xlwt
#from gru_cus_cell import EGGRUCell
#from gru_poly_cell import GRUPolyCell

tqdm.monitor_interval = 0
#from utils_2 import np_REG_batch, search_wav, wav2spec, spec2wav, copy_file, np_batch, get_embedding, get_dist_table
from utils import np_REG_batch, search_wav, copy_file, np_batch, get_embedding, get_dist_table, _gen_audio
from sklearn.utils import shuffle
import tensorflow_utils as tfu
import wandb

#eps = np.finfo(np.float32).epsilon()

class REG:

    def __init__(self, log_path, saver_path, date, gpu_num, note, config):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        self.log_path = log_path
        self.saver_path = saver_path
        self.saver_dir = '{}_{}/{}'.format( self.saver_path, note, date )
        self.eps = np.finfo(np.float32).eps
        self.saver_name = join(
            self.saver_dir, 'best_saver_{}'.format( note ) )
        self.tb_dir = '{}_{}/{}'.format( self.log_path, note, date )
        self.config = config

        if not os.path.exists( self.saver_dir ):
            os.makedirs( self.saver_dir )
        if not os.path.exists( self.tb_dir ):
            os.makedirs( self.tb_dir )


    def build(self, reuse):

        wandb.init(project="VAD", entity="musktang")
        # wandb.config = {
        #     "batch_size" : self.config.batch_size,
        #     "filter_h" : 3,
        #     "filter_w" : 2,
        #     "mel_freq_num" : self.config.mel_freq_num,
        #     "l1_output_num" : 20,
        #     "l2_output_num": 10,
        #     "l3_output_num": 30,
        # }
        wandb.config = {
            "batch_size": self.config.batch_size,
            "filter_h": 5,
            "filter_w": 5,
            "mel_freq_num": self.config.mel_freq_num,
            "l1_output_num": 40,
            "l2_output_num": 20,
            "l3_output_num": 10,
        }

        self.name = 'VAD'
        input_dimension = 24  # RNN input
        output_dimension = 1

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            with tf.variable_scope('Intputs'):
                self.x_noisy_norm = tf.placeholder(
                    tf.float32, shape=[None, input_dimension, self.config.stoi_correlation_time, 1], name='x_norm')

                self.lr = tf.placeholder(dtype=tf.float32)  # learning rate
                self.keep_prob = 0.7
            with tf.variable_scope('Outputs'):
                self.ground_truth = tf.placeholder(
                    tf.float32, shape=[None, self.config.stoi_correlation_time, 1], name='ground_truth')

            with tf.variable_scope('featureExtractor', reuse=tf.AUTO_REUSE):
                layer_1 = tfu._add_conv_layer(self.x_noisy_norm, layer_num='1', filter_h=wandb.config.get("filter_h"),
                                              filter_w=wandb.config.get("filter_w"), input_c=1,
                                               output_c=wandb.config.get("l1_output_num"), dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME', trainable=True)  # [N, 126, t-2, 512]
                layer_2 = tfu._add_conv_layer(layer_1, layer_num='2', filter_h=wandb.config.get("filter_h"),
                                              filter_w=wandb.config.get("filter_w"), input_c=wandb.config.get("l1_output_num"),
                                              output_c=wandb.config.get("l2_output_num"), dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME', trainable=True)  # [N, 62, t-4, 512]
                layer_3 = tfu._add_conv_layer(layer_2, layer_num='3', filter_h=wandb.config.get("filter_h"),
                                              filter_w=1, input_c=wandb.config.get("l2_output_num"),
                                              output_c=wandb.config.get("l3_output_num"), dilation=[1, 1, 1, 1],
                                              activate=tf.nn.leaky_relu, padding='SAME', trainable=True)  # [N, 124, t-4, 128]
                reshape = tf.reshape(tf.transpose(layer_3, perm=[0, 2, 3, 1]),
                                     [-1, self.config.stoi_correlation_time, wandb.config.get("l3_output_num") * input_dimension])
                output = tfu._add_3dfc_layer(reshape, wandb.config.get("l3_output_num") * input_dimension, 1,
                                               '4', activate_function=tf.nn.sigmoid, trainable=True, keep_prob=1)
                # softmax = tf.nn.softmax(layer_4, )

            with tf.name_scope('reg_loss'):
                self.loss_mse_denoiser = tf.losses.mean_squared_error(output, self.ground_truth)
                predict_speech = tf.cast(output>0.5, tf.float32)
                self.speech_hit_rate = tf.div(tf.reduce_sum(tf.multiply(predict_speech,self.ground_truth)),tf.reduce_sum(self.ground_truth))
                self.noise_hit_rate = tf.div(tf.reduce_sum(tf.multiply(tf.subtract(1.0, predict_speech),tf.subtract(1.0, self.ground_truth)))
                                             , tf.reduce_sum(tf.subtract(1.0, self.ground_truth)))
                self.total_loss = self.loss_mse_denoiser
                tf.summary.scalar('Loss mse', self.loss_mse_denoiser)
                tf.summary.scalar('SHR', self.speech_hit_rate)
                tf.summary.scalar('NHR', self.noise_hit_rate)
                # wandb.log({"Loss mse": self.loss_mse_denoiser})


            with tf.name_scope("exp_learning_rate"):
                self.global_step = tf.Variable(0, trainable=False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.lr)
                gradients_1, v_1 = zip(*optimizer.compute_gradients(self.total_loss))
                self.optimizer_1 = optimizer.apply_gradients(zip(gradients_1, v_1),
                                                             global_step=self.global_step)
            var_list = tf.all_variables()

            # self.saver_pre1 = tf.train.Saver(
            #     var_list=[v for v in var_list if 'modual1' in v.name or 'featureExtractor' in v.name])
            # self.saver_pre2 = tf.train.Saver(
            #     var_list=[v for v in var_list if 'modual2' in v.name in v.name])
            self.saver = tf.train.Saver()

    def _training_process(self, sess, epoch, data_list, noise_list, snr_list, merge_op, step, writer, learning_rate, train=True):
                #self.reg_layer = mask*tf.reshape(self.x_noisy[:,:, 1:-1, :], [-1, self.config.stoi_correlation_time, output_dimension])
                #self.reg_layer = mask*tf.reshape(self.x_noisy[:,:, 1:-1, :], [-1, self.config.stoi_correlation_time, output_dimension])
      
        loss_reg_tmp = 0.
        count = 0
        audio_len = len( data_list )
        learning_rate = learning_rate
        
        audio_batch_size = self.config.audio_batch
        get_audio_batch = np_batch(
            data_list, audio_batch_size, audio_len )
        get_noise_batch = np_batch(
            noise_list, audio_batch_size, audio_len )

        for audio_iteration in tqdm(range(int( audio_len / audio_batch_size ))):
            audio_batch = next(get_audio_batch)
            noise_batch = next(get_noise_batch)
            pool = Pool(processes=self.config.thread_num)       
            func = partial( _gen_VAD_training_data_runtime, audio_batch, noise_batch,
                            self.config, train)
            training_data = pool.map( func, range( 0, audio_batch_size ) )
            pool.close()
            pool.join()

            dim = 0
            training_data_trans = zip(*training_data)
            for data in training_data_trans:
                if dim == 0:
                    noisy_norm_data = np.vstack( data )
                if dim == 1:
                    ground_truth = np.vstack( data )
                # if dim == 2:
                #     noisy_data_norm = np.vstack( data )
                # if dim == 3:
                #     clean_data_norm = np.vstack( data )
                dim += 1
            del training_data, training_data_trans

            noisy_norm_data, ground_truth = shuffle( noisy_norm_data, ground_truth)
            #t0 = time.time()
            data_len = len( noisy_norm_data )
            data_batch = np_REG_batch(
                noisy_norm_data, ground_truth, self.config.batch_size, data_len)
            for batch in range( int( data_len / self.config.batch_size ) ):
                noisy_norm_batch, ground_truth_batch= next(
                    data_batch ), next( data_batch )
        
                if train:
                    keep_prob = 0.7
                else:
                    keep_prob = 1
                feed_dict = {
                             self.lr: learning_rate,
                             self.x_noisy_norm: noisy_norm_batch,
                             self.ground_truth: ground_truth_batch,
                             #self.train : train
                             }
                if train:
                    _, loss_reg, summary, SHR, NHR = sess.run(
                        [self.optimizer_1, self.total_loss, merge_op, self.speech_hit_rate, self.noise_hit_rate
                         ], feed_dict=feed_dict )
                    
                    step += 1
                    writer.add_summary( summary, step )
                    wandb.log({"train mse loss": loss_reg})
                    wandb.log({"train SHR": SHR})
                    wandb.log({"train NHR": NHR})
                else:
                    loss_reg, summary, SHR, NHR = sess.run(
                        [self.total_loss, merge_op, self.speech_hit_rate, self.noise_hit_rate
                         ], feed_dict=feed_dict )
                    wandb.log({"dev mse loss": loss_reg})
                    wandb.log({"dev SHR": SHR})
                    wandb.log({"dev NHR": NHR})
                
                loss_reg_tmp += loss_reg
                count += 1
            #print('GPU processing time :' + str(time.time()-t0))
        #print('count = ', count)
        loss_reg_tmp /= count
        if not train:
            writer.add_summary( summary, step )

        #if not self.config.boost:
        return loss_reg_tmp, summary, step
        #else:
        #    return loss_reg_tmp, summary, step, audio_loss


    def train(self, read_ckpt=None):
        if tf.gfile.Exists( self.tb_dir ):
            tf.gfile.DeleteRecursively( self.tb_dir )
            tf.gfile.MkDir( self.tb_dir )

        best_dev_loss = 10.

        data_list = [tag for tag in iglob( self.config.voice_path )]
        # mozilla_data_list = [tag for tag in iglob(self.config.mozilla_train_path)]
        # librispeech_data_list = [tag for tag in iglob(self.config.librispeech_train_path)]
        # vocalset_data_list = [tag for tag in iglob(self.config.vocalset_train_path)]
        # data_list.extend(mozilla_data_list)
        # data_list.extend(librispeech_data_list)
        # data_list.extend(vocalset_data_list)

        noise_list = [tag for tag in iglob( self.config.noise_path )]
        # audioset_noise_list = [tag for tag in iglob(self.config.audioset_path)]
        # noise_list.extend(audioset_noise_list)

        noise_list_tmp = np.random.choice(noise_list, len(data_list))


        dev_data_list = [tag for tag in iglob( self.config.dev_voice_path )]
        # mozilla_data_list = [tag for tag in iglob(self.config.mozilla_dev_path)]
        # librispeech_data_list = [tag for tag in iglob(self.config.librispeech_dev_path)]
        # vocalset_data_list = [tag for tag in iglob(self.config.vocalset_dev_path)]
        # dev_data_list.extend(mozilla_data_list)
        # dev_data_list.extend(librispeech_data_list)
        # dev_data_list.extend(vocalset_data_list)

        dev_noise_list = [tag for tag in iglob( self.config.dev_noise_path )]
        dev_noise_list = np.random.choice(dev_noise_list, len(dev_data_list))

        with tf.Session() as sess:

            print( 'Start Training' )
            # set early stopping
            patience = 4
            FLAG = True
            learning_rate = self.config.init_learning_rate
            min_delta = self.config.min_delta_loss
            step = 0
            epochs = range( self.config.epochs )

            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(
                self.tb_dir+'/train', sess.graph, max_queue=10 )
            validation_writer = tf.summary.FileWriter(self.tb_dir + '/validation', sess.graph)
            merge_op = tf.summary.merge_all()

            # wandb.tensorflow.log(tf.summary.merge_all())
            # self.saver.restore(sess=sess, save_path=test_saver)
            
            ####################    Musk    ###################################

            if read_ckpt is not None:
                model_path = "/AudioProject/nb_denoise/model/" + read_ckpt + "/"
                # model_path1 = "/AudioProject/nb_denoise/model/saver_module_model_humanNoise/210524-1/"
                model_path2 = "/AudioProject/nb_denoise/model/saver_moduleModel/220105-regular/"
                ckpt1 = tf.train.get_checkpoint_state(model_path)
                ckpt2 = tf.train.get_checkpoint_state(model_path2)
                if ckpt1 is not None:
                    print("Model path : " + ckpt1.model_checkpoint_path)
                    # print("Model path : " + ckpt2.model_checkpoint_path)
                    self.saver_pre1.restore(sess, ckpt1.model_checkpoint_path)
                    self.saver_pre2.restore(sess, ckpt2.model_checkpoint_path)
                else:
                    print("model not found")

            ###################################################################
            snr = ['-2dB', '-1dB', '-5dB']
            audio_len = len( data_list )
            noise_len = len( noise_list )
            snr_list = np.random.choice(snr, audio_len)
            # with open('train_snr.csv', 'w') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerows(snr_list)

            dev_snr = ['-4dB']
            dev_audio_len = len( dev_data_list )
            dev_snr_list = np.random.choice( dev_snr, dev_audio_len )

            ###################################################################
            
            for epoch in tqdm( epochs ):

                if epoch % self.config.shuffle_data_time == 0 :
                    data_list = shuffle( data_list )
                    noise_list_tmp = np.random.choice(noise_list, len(data_list))
                    snr_list = np.random.choice(snr, audio_len)

                loss_reg, summary, step = self._training_process(sess, epoch, data_list,
                                                                 noise_list_tmp, snr_list, 
                                                                 merge_op, step, writer, learning_rate, train=True)

                loss_dev, summary_dev, _ = self._training_process(sess, epoch, dev_data_list,
                                                                 dev_noise_list, dev_snr_list,
                                                                 merge_op, step, validation_writer, learning_rate, train=False)

                # wandb.log({"train mse loss" : loss_reg, "dev mse loss" : loss_dev})

                if epoch == 0:
                    best_reg_loss = loss_dev
                    print( '[epoch {}] Loss reg:{}'.format(
                        int( epoch ), loss_reg ) )
                    print( '[epoch {}] Loss Dev:{}'.format(
                        int( epoch ), loss_dev ) )
                    self.saver.save( sess=sess, save_path=self.saver_name )
                else:
                    print( '[epoch {}] Loss reg:{}'.format(
                        int( epoch ), loss_reg ) )
                    print( '[epoch {}] Loss Dev:{}'.format(
                        int( epoch ), loss_dev ) )

                    if loss_dev <= (best_reg_loss - min_delta):
                        best_reg_loss = loss_dev
                        self.saver.save( sess=sess, save_path=self.saver_name )
                        patience = 4
                        print( 'Best Reg Loss: ', best_reg_loss )
                    else:
                        print( 'Not improve Loss:', best_reg_loss )
                        if FLAG == True:
                            patience -= 1
                if patience == 0 and FLAG == True:
                    learning_rate /= 2
                    if learning_rate < 1e-4:
                        learning_rate *= 10
                    patience = 4
                    print( 'Learning Rate Decrease ! ! ! : {}'.format(learning_rate) )
