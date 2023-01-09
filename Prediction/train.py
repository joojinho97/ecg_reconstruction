from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping)
from ResNet.resnet import get_model

if __name__ == "__main__":
    # Get data and train
    x_train = "x_train"
    y_train = "y_train"
    x_vali = "x_vali"
    y_vali = "y_vali"

    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    model = get_model(3)
    model.compile(loss=loss, optimizer=opt)

    history = model.fit(x_train, y_train, validation_data=(x_vali, y_vali),
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        verbose=2)

    model.save("./model.hdf5")
