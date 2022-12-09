# @Author: Yu Zhaoke  
# @Date: 2021-09-27 03:14:31  
# @Last Modified by:   Your name
# @Last Modified time: 2021-10-14 03:05:54


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import LayerNormalization


class kz_model(keras.Model):
    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        self.action_dim = self.config['action_dim']
        self.split_list = self.config['split_list']
        self.vnet_outdim = self.config['vnet_outdim']

        self.fc_control1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.fc_control2 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')

        self.fc_my1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc_my2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.max_my = layers.MaxPool2D((5, 1), strides=1, data_format='channels_last')

        self.fc_opp1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc_opp2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.max_opp = layers.MaxPool2D((5, 1), strides=1, data_format='channels_last')

        self.fc_my_missile1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc_my_missile2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.max_my_missile = layers.MaxPool2D((12, 1), strides=1, data_format='channels_last')

        self.fc_opp_missile1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc_opp_missile2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.max_opp_missile = layers.MaxPool2D((12, 1), strides=1, data_format='channels_last')

        self.fc1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc1_norm = LayerNormalization()

        self.fc2 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.fc2_norm = LayerNormalization()

        self.vnet1 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.vnet1_norm = LayerNormalization()
        # self.d1=layers.Dropout(0.5)
        self.vnet2 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.vnet2_norm = LayerNormalization()
        # self.d1=layers.Dropout(0.5)
        self.vnet3 = layers.Dense(self.vnet_outdim, activation=None, kernel_initializer='he_uniform')

        self.output1 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.output1_norm = LayerNormalization()
        self.output2 = layers.Dense(self.action_dim, activation=None, kernel_initializer='he_uniform')

        self.set_batchsize(1)
        self.debug = debug

    def set_batchsize(self, batch_size):
        self.batch_size = batch_size

    def call(self, inputs):
        # print(f'----------------{inputs.shape}')
        control_in, my_in, opp_in, my_missile_in, opp_missile_in = tf.split(inputs, self.split_list, axis=-1)
        # control_in: none, 5, 14
        # my_in: none, 5, 22*5
        my_in = tf.stack(tf.split(my_in, 5, axis=-1), axis=-2)  # none, n, m, x -> none, 5, 5, 22
        opp_in = tf.stack(tf.split(opp_in, 5, axis=-1), axis=-2)  # none, n, m, x -> none, 5, 5, 20
        my_missile_in = tf.stack(tf.split(my_missile_in, 12, axis=-1), axis=-2)  # none, n, m, x -> none, 5, 12, 35
        opp_missile_in = tf.stack(tf.split(opp_missile_in, 12, axis=-1), axis=-2)  # none, n, m, x -> none, 5, 12, 35

        control_embed = self.fc_control1(control_in)  # none, 5, 512
        control_embed = self.fc_control2(control_embed)  # none, 5, 32

        # print(f'----------------{my_in.shape}')

        my_embed = self.fc_my1(my_in)  # none, 5, 5, 1024
        my_embed = self.fc_my2(my_embed)  # none, 5, 5, 64
        my_embed = tf.transpose(my_embed, (0, 2, 3, 1))  # none, 5, 64, 5
        my_embed = tf.squeeze(self.max_my(my_embed), axis=1)  # none, 64, 5
        my_embed = tf.transpose(my_embed, (0, 2, 1))  # none, 5, 64

        opp_embed = self.fc_opp1(opp_in)  # none, 5, 5, 1024
        opp_embed = self.fc_opp2(opp_embed)  # none, 5, 5, 64
        opp_embed = tf.transpose(opp_embed, (0, 2, 3, 1))  # none, 5, 64, 5
        # print(f'----------------{opp_embed.shape}')
        opp_embed = tf.squeeze(self.max_opp(opp_embed), axis=1)  # none, 64, 5
        opp_embed = tf.transpose(opp_embed, (0, 2, 1))  # none, 5, 64
        # print(f'----------------{opp_embed.shape}')

        my_missile_embed = self.fc_my_missile1(my_missile_in)  # none, 5, 12, 1024
        my_missile_embed = self.fc_my_missile2(my_missile_embed)  # none, 5, 12, 64
        my_missile_embed = tf.transpose(my_missile_embed, (0, 2, 3, 1))  # none, 12, 64, 5
        my_missile_embed = tf.squeeze(self.max_my_missile(my_missile_embed), axis=1)  # none, 64, 5
        my_missile_embed = tf.transpose(my_missile_embed, (0, 2, 1))  # none, 5, 64

        opp_missile_embed = self.fc_opp_missile1(opp_missile_in)  # none, 5, 12, 1024
        opp_missile_embed = self.fc_opp_missile2(opp_missile_embed)  # none, 5, 12, 64
        opp_missile_embed = tf.transpose(opp_missile_embed, (0, 2, 3, 1))  # none, 12, 64, 5
        opp_missile_embed = tf.squeeze(self.max_opp_missile(opp_missile_embed), axis=1)  # none, 64, 5
        opp_missile_embed = tf.transpose(opp_missile_embed, (0, 2, 1))  # none, 5, 64
        # print(f'----------------{opp_missile_embed.shape}')

        cat = tf.concat([control_embed, my_embed, opp_embed, my_missile_embed, opp_missile_embed],
                        axis=-1)  # none, 5, 32+64*4
        # print(f'----------------{cat.shape}')
        cat = self.fc1(cat)  # none, 5, 1024
        cat = self.fc1_norm(cat)
        cat = self.fc2(cat)  # none, 5, 1024
        cat = self.fc2_norm(cat)

        v_out = self.vnet1(cat)  # none, 5, 1024
        v_out = self.vnet1_norm(v_out)
        v_out = self.vnet2(v_out)  # none, 5, 512
        v_out = self.vnet2_norm(v_out)
        v_out = self.vnet3(v_out)  # none, 5, 1
        # print(f'----------------{v_out.shape}')
        v_out = tf.squeeze(v_out, axis=2)  # none, 5

        logits = self.output1(cat)  # none, 5, 512
        logits = self.output1_norm(logits)
        logits = self.output2(logits)  # none, 5, 6
        logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)

        return v_out, logits