import numpy as np
from metric import *
from Two_step_VAE.onedvae import load_onedvae
from Two_step_VAE.vae import load_vae as twodvae
from Pix2Pix.pix2pix import load_generator as pix2pix_generator
from EKGAN.ekgan import load_inference_generator as ekgan_generator
from CycleGAN.cyclegan import load_generator as cyclegan_generator
from CardioGAN.cardiogan import load_generator as cardiogan_generator

'''
test code
we evalutate each model using RMSE,MAE,PRD.
'''


def test(tdata, tdata_label, model_name, path):
    if model_name == '1':
        model = ekgan_generator()
        model.load_weights(f'{path}/generator10.h5')
    elif model_name == '2':
        model = pix2pix_generator()
        model.load_weights(f'{path}/generator10.h5')
    elif model_name == '3':
        model = cyclegan_generator()
        model.load_weights(f'{path}/generator10.h5')
    elif model_name == '4':
        print('a')
        model = cardiogan_generator()
        model.load_weights(f'{path}/generator10.h5')
    elif model_name == '5':
        model = twodvae()
        model.load_weights(f'{path}/generator150.h5')

    predict_result = model(tdata, training=False)

    t = np.arange(0., 512., 1)

    print('result ======')
    calculate_metric(tdata_label, predict_result, tdata.shape[0], model)


if __name__ == '__main__':
    '''
        model name =
            1 : EKGAN
            2 : Pix2Pix
            3 : CycleGAN
            4 : CardioGAN
            5 : VAE
    '''
    model_name = 'write your model number'
    path = 'write your model weight directory'

    test(data, data_label, model_name, path)
