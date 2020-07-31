import tensorflow as tf
from tensorflow.python.client import device_lib


if __name__ == "__main__":
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess)

    print(device_lib.list_local_devices())
    print('----------------------')

    from keras import backend as K
    print(K.tensorflow_backend._get_available_gpus())
