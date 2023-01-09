import tensorflow as tf
import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import EKGAN.ekgan as ekgan
import Pix2Pix.pix2pix as pix2pix
import CycleGAN.cyclegan as cyclegan
import CardioGAN.cardiogan as cardiogan
import Two_step_VAE.onedvae as onedvae
import Two_step_VAE.vae as vae
from Two_step_VAE.loss import *

'''
training code

each function shows model training pipeline for experiment result in our paper
train_ekgan : EKGAN for Lead I-> 12 Lead ECG
train_pix2pix : Pix2Pix for Lead I -> 12 Lead ECG
train cyclegan : CycleGAN for Lead I -> 12 Lead ECG
train_cardiogan : CardioGAN for Lead I -> 12 Lead ECG
train_two_step_onedvae for Lead I -> Lead II
train_two_step_twodvae : VAE for 6 Limb Lead(aVL,Lead I,-aVR,Lead II, aVF, Lead III) -> 6Precordial Lead(V1~V6)
'''


# learning rate decay
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, _len):
        self.initial_learning_rate = initial_learning_rate
        self.data_len = _len

    def __call__(self, step):
        if (step < int(self.data_len*5)):
            return self.initial_learning_rate
        elif step % int(self.data_len) == 0:
            return self.initial_learning_rate*0.95
        else:
            return self.initial_learning_rate


def train_ekgan(train, train_label, path):
    ''' EKGAN '''
    epochs = 10
    batch_size = 32
    _lambda = 50

    inference_generator = ekgan.load_inference_generator()
    label_generator = ekgan.load_label_generator()
    discriminator = ekgan.load_discriminator()

    label_generator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    inference_generator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i in range(int(len(train)/32)-1):
            tr = train[batch_size*i:batch_size*(i+1)]
            tr_label = train_label[batch_size*i:batch_size*(i+1)]
            ekgan.train_step(tr, tr_label, inference_generator, discriminator,
                             inference_generator_optimizer, discriminator_optimizer, epoch,
                             label_generator, label_generator_optimizer, _lambda)
        if (epoch + 1) % 1 == 0:
            inference_generator.save_weights(f'{path}/EKGAN/generator{epoch+1}.h5')


def train_pix2pix(train, train_label, path):
    ''' Pix2Pix '''
    epochs = 10
    batch_size = 32
    _lambda = 50

    generator = pix2pix.load_generator()
    discriminator = pix2pix.load_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i in range(int(len(train)/32)-1):
            tr = train[batch_size*i:batch_size*(i+1)]
            tr_label = train_label[batch_size*i:batch_size*(i+1)]
            pix2pix.train_step(tr, tr_label, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs, _lambda)

        if (epoch + 1) % 1 == 0:
            generator.save_weights(f'{path}/Pix2Pix/generator{epoch+1}.h5')


def train_cyclegan(train, train_label, path):
    '''cyclegan'''
    epochs = 10
    batch_size = 32
    generator_x = cyclegan.load_generator()
    discriminator_x = cyclegan.load_discriminator()

    generator_y = cyclegan.load_generator()
    discriminator_y = cyclegan.load_discriminator()

    generator_x_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    generator_y_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    for epoch in range(epochs):

        print("Epoch: ", epoch)
        for i in range(int(len(train)/32)-1):
            tr = train[batch_size*i:batch_size*(i+1)]
            tr_label = train_label[batch_size*i:batch_size*(i+1)]
            cyclegan.train_step(tr, tr_label, generator_x, discriminator_x, generator_x_optimizer,
                                discriminator_x_optimizer, epoch, generator_y, generator_y_optimizer,
                                discriminator_y, discriminator_y_optimizer)
        if (epoch + 1) % 1 == 0:
            generator_x.save_weights(f'{path}/CycleGAN/generator{epoch+1}.h5')


def train_cardiogan(train, train_label, path):
    '''cardiogan'''
    epochs = 10
    batch_size = 32

    generator_g = cardiogan.load_generator()
    discriminator_g = cardiogan.load_time_discriminator()
    discriminator_g2 = cardiogan.load_time_discriminator()

    generator_f = cardiogan.load_generator()
    discriminator_f = cardiogan.load_frequency_discriminator()
    discriminator_f2 = cardiogan.load_frequency_discriminator()

    generator_g_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)

    discriminator_g_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_g2_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_f_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)
    discriminator_f2_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(1e-4, len(train)), beta_1=0.5)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i in range(int(len(train)/32)-1):
            tr = train[batch_size*i:batch_size*(i+1)]
            tr_label = train_label[batch_size*i:batch_size*(i+1)]
            cardiogan.train_step(tr, tr_label, generator_g, generator_f,
                                 discriminator_g, discriminator_f, generator_g_optimizer,
                                 generator_f_optimizer, discriminator_g_optimizer, discriminator_f_optimizer,
                                 epoch, discriminator_g2, discriminator_f2, discriminator_g2_optimizer, discriminator_f2_optimizer)
    if (epoch + 1) % 1 == 0:
        generator_g.save_weights(f'{path}/CardioGAN/generator{epoch+1}.h5')


def train_two_step_onedvae(train, train_label, path):
    '''one stage vae'''
    oned_vae = onedvae.load_onedvae()
    oned_vae.compile(loss=[mse], optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[mse_keras])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'{path}/onedvae_generator_{epoch:02d}.h5', verbose=True, save_weights_only=True, mode='auto')]
    history = oned_vae.fit(train, train_label, epochs=150, verbose=1)


def train_two_step_vae(train, train_label, path):
    '''two stage vae'''
    twod_vae = vae.load_vae()

    twod_vae.compile(loss=[mse], optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[mse_keras])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'{path}/twodvae_generator_{epoch:02d}.h5', verbose=True, save_weights_only=True, mode='auto')]
    history = twod_vae.fit(data, data_label, epochs=150, verbose=1)


if __name__ == '__main__':
    ''' data : train dataset, data_label : target label '''
    # path='write your dataset directory'
    # train_ekgan(data, data_label, path)
    # train_pix2pix(data, data_label, path)
    # train_cyclegan(data, data_label, path)
    # train_cardiogan(data, data_label, path)
    # train_two_step_onedvae(data, data_label, path)
    # train_two_step_vae(data, data_label, path)
