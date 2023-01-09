import tensorflow as tf
import keras
from tensorflow.keras import layers
from EKGAN.loss import *

def load_inference_generator():

    '''
    inference_generator

    inference generator is based on U-net architecture.
    inference generator take (16,512,1) data and final output is (16,512,1) too.
    input and output data contain two row zero padding on the top and bottom
    using 5 block in encoder and decoder, it contains 1 convolution layer each block.
    batchnormalization and leaky-ReLu use after convolution layer in encoder , except first block
    decoder use Relu.

    initializer : initializer of each layer, random choise in (mean =0, stdev = 0.02)
    encoder_inputs : Input of inference generator
    number_of_filter_encoder : filter of encoder in inference generator
    number_of_filter_decoder : filter of decoder in inference generator
    kernel : kernel
    stride : stride
    '''

    initializer = tf.random_normal_initializer(0., 0.02)
    encoder_inputs = keras.Input(shape=(16, 512, 1), name='inference_generator_input')
    number_of_filter_encoder = [64, 128, 256, 512, 1024]
    number_of_filter_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]

    concatenate_encoder_block = []
    b_conv = tf.keras.layers.Conv2D(64, kernel, strides=stride[0], padding='same', kernel_initializer=initializer, use_bias=False)(encoder_inputs)
    b_output = tf.keras.layers.Activation('LeakyReLU')(b_conv)
    concatenate_encoder_block.append(b_output)

    for i in range(1, 5, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], kernel, strides=stride[i], padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('LeakyReLU')(b_batch)
        if i == 4:
            break
        concatenate_encoder_block.append(b_output)

    b_conv = tf.keras.layers.Conv2DTranspose(512, (2, 4), strides=(1, 2), padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
    b_batch = tf.keras.layers.BatchNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 4, 1):
        b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-i]])
        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], kernel, strides=stride[4-i], padding='same', kernel_initializer=initializer, use_bias=False)(b_concat)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-4]])
    encoder_outputs = layers.Conv2DTranspose(1, (2, 4), strides=(2), padding="same", kernel_initializer=initializer, use_bias=False)(b_concat)

    return keras.Model(encoder_inputs, encoder_outputs)


def load_label_generator():

    '''
    label_generator

    label generator is almost same as inference generator.
    but label generator concatenate generated data from inference generator and target label in front of first block.
    label generator take (16,512,2) data and final output is (16,512,1).
    input and output data contain two row zero padding on the top and bottom
    using 5 block in encoder and decoder, it contains 1 convolution layer each block.
    batchnormalization and leaky-ReLu use after convolution layer in encoder , except first block
    decoder use Relu.
    initializer : initializer of each layer, random choise in (mean =0, stdev = 0.02)
    encoder_inputs : Input of inference generator
    number_of_filter_encoder : filter of encoder in inference generator
    number_of_filter_decoder : filter of decoder in inference generator
    kernel : kernel
    stride : stride
    '''

    initializer = tf.random_normal_initializer(0., 0.02)
    encoder_inputs = keras.Input(shape=(16, 512, 1), name='generated_inference_generator')
    tar = keras.Input(shape=(16, 512, 1), name='target')

    number_of_filter_encoder = [64, 128, 256, 512, 1024]
    number_of_filter_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]
    concatenate_encoder_block = []

    concat = tf.keras.layers.Concatenate()([encoder_inputs, tar])
    b_conv = tf.keras.layers.Conv2D(64, kernel, strides=stride[0], padding='same', kernel_initializer=initializer, use_bias=False)(concat)
    b_output = tf.keras.layers.Activation('LeakyReLU')(b_conv)
    concatenate_encoder_block.append(b_output)

    for i in range(1, 5, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], kernel, strides=stride[i], padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('LeakyReLU')(b_batch)
        if i == 4:
            break
        concatenate_encoder_block.append(b_output)

    b_conv = tf.keras.layers.Conv2DTranspose(512, (2, 4), strides=(1, 2), padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
    b_batch = tf.keras.layers.BatchNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 4, 1):
        b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-i]])
        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], kernel, strides=stride[4-i], padding='same', kernel_initializer=initializer, use_bias=False)(b_concat)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-4]])
    encoder_outputs = layers.Conv2DTranspose(1, (2, 4), strides=(2), padding="same", kernel_initializer=initializer, use_bias=False)(b_concat)

    return tf.keras.Model(inputs=[encoder_inputs, tar], outputs=encoder_outputs)


def load_discriminator():

    '''
    discriminator

    discriminator is almost same as label generator.
    discriminator concatenate input data and generated data from inference generator / label generator in front of first block.
    label generator take (16,512,2) data and final output is (16,512,1) too.
    discriminator use 1d-convolution.
    using 5 block in encoder and decoder, it contains 1 convolution layer each block.
    batchnormalization and leaky-ReLu use after convolution layer in encoder , except first block
    decoder use Relu.

    initializer : initializer of each layer, random choise in (mean =0, stdev = 0.02)
    encoder_inputs : Input of inference generator
    number_of_filter_encoder : filter of encoder in inference generator
    number_of_filter_decoder : filter of decoder in inference generator
    kernel : kernel
    stride : stride
    '''

    initializer = tf.random_normal_initializer(0., 0.02)

    encoder_inputs = keras.Input(shape=(16, 512, 1), name='input_image')
    tar = keras.Input(shape=(16, 512, 1), name='generated from label_generator or inference_generator')
    number_of_filter_encoder = [32, 64, 128, 256, 512]
    number_of_filter_decoder = [256, 128, 64, 32, 1]
    kernel = [64, 32, 16, 8, 4]
    stride = [4, 4, 4, 2, 2]
    concatenate_encoder_block = []

    concat = tf.keras.layers.Concatenate()([encoder_inputs, tar])
    b_conv = tf.keras.layers.Conv2D(32, (1, kernel[0]), strides=(1, stride[0]), padding='same', kernel_initializer=initializer, use_bias=False)(concat)
    b_output = tf.keras.layers.Activation('LeakyReLU')(b_conv)
    concatenate_encoder_block.append(b_output)

    for i in range(1, 5, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], (1, kernel[i]), strides=(1, stride[i]), padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('LeakyReLU')(b_batch)
        if i == 4:
            break
        concatenate_encoder_block.append(b_output)

    b_conv = tf.keras.layers.Conv2DTranspose(256, (1, kernel[-1]), strides=(1, 2), padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
    b_batch = tf.keras.layers.BatchNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 4, 1):
        b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-i]])
        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], (1, kernel[4-i]), strides=(1, stride[4-i]), padding='same', kernel_initializer=initializer, use_bias=False)(b_concat)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-4]])
    encoder_outputs = layers.Conv2DTranspose(1, (1, 64), strides=(1, 4), padding="same", kernel_initializer=initializer, use_bias=False)(b_concat)

    return tf.keras.Model(inputs=[encoder_inputs, tar], outputs=encoder_outputs)


def train_step(input_image, target, inference_generator, discriminator, inference_generator_optimizer, discriminator_optimizer, epoch, label_generator, label_generator_optimizer, _lambda):
    with tf.GradientTape() as inference_gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as label_gen_tape:
        inference_gen_output = inference_generator(input_image, training=True)

        label_gen_output = label_generator([inference_gen_output, target], training=True)
        disc_real_output = discriminator([input_image, label_gen_output], training=True)
        disc_generated_output = discriminator([input_image, inference_gen_output], training=True)

        total_label_generator_loss = label_generator_loss(target, label_gen_output)
        total_inference_generator_loss, inference_gen_gan_loss, inference_gen_l1_loss = inference_generator_loss(disc_generated_output, inference_gen_output, target, _lambda)
        total_discriminator_loss = discriminator_loss(disc_real_output, disc_generated_output)

    inference_generator_gradients = inference_gen_tape.gradient(total_inference_generator_loss, inference_generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_discriminator_loss, discriminator.trainable_variables)
    label_generator_gradients = label_gen_tape.gradient(total_label_generator_loss, label_generator.trainable_variables)

    inference_generator_optimizer.apply_gradients(zip(inference_generator_gradients, inference_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    label_generator_optimizer.apply_gradients(zip(label_generator_gradients, label_generator.trainable_variables))

    print('epoch {} total_inference_generator_loss {} inference_gen_gan_loss {} inference_gen_l1_loss {} total_label_generator_loss {}'.format(epoch, total_inference_generator_loss, inference_gen_gan_loss, inference_gen_l1_loss, total_label_generator_loss))
