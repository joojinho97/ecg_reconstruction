import numpy as np
import tensorflow as tf

def mae(y_true, y_pred):
    return np.abs(np.mean(np.subtract(y_pred, y_true), axis=-1))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))


def prd(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2) / np.sum(y_true**2) * 100)


def MAE_(y_true, y_pred):
    return np.mean(np.abs(y_true) * np.abs(y_true - y_pred))


def calculate_metric(y_true, y_pred, data_len, generator, generator2=None):
    y_true = np.delete(y_true, 9, axis=1)
    y_pred = np.delete(y_pred, 9, axis=1)

    result = 0
    for j in range(data_len):
        RMSE = tf.keras.metrics.MeanSquaredError()
        RMSE.update_state(y_true[j][2:13, :, :], y_pred[j][2:13, :, :])
        result += np.sqrt(RMSE.result().numpy())
    print('RMSE keras :', result / data_len)
                                                                                         
    result = 0
    for j in range(data_len):
        result += rmse(y_true[j][2:13, :, :], y_pred[j][2:13, :, :])
    print('rmse :', result / data_len)

    result = 0
    for j in range(data_len):
        MAE = tf.keras.metrics.MeanAbsoluteError()
        MAE.update_state(y_true[j][2:13, :, :], y_pred[j][2:13, :, :])
        result += MAE.result().numpy()
    print('MAE :', result / data_len)

    result = 0
    for j in range(data_len):
        result += prd(y_true[j][2:13, :, :], y_pred[j][2:13, :, :])
    print('prd :', result / data_len)

    result = 0
    result_MAE = 0
    for i in range(11):
        result = 0
        result_MAE = 0
        result_prd = 0
        for j in range(data_len):
            RMSE = tf.keras.metrics.MeanSquaredError()
            RMSE.update_state(y_true[j][i+2, :, :], y_pred[j][i+2, :, :])
            result += np.sqrt(RMSE.result().numpy())

            MAE = tf.keras.metrics.MeanAbsoluteError()
            MAE.update_state(y_true[j][i+2, :, :], y_pred[j][i+2, :, :])
            result_MAE += MAE.result().numpy()

            result_prd += prd(y_true[j][i+2, :, :], y_pred[j][i+2, :, :])

        print('RMSE result :', result / data_len)
        print('MAE result :', result_MAE / data_len)
        print('prd result :', result_prd / data_len)
