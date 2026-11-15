import tensorflow as tf
from tensorflow.nn import conv2d

def SB_ATV(x, mu, z=None,c = 3, alpha=0.002, n_iter=5):
    """Solves the SB_ATV denoising problem using the split Bregman algorithm.

    Args:
        x (tf.Tensor): The noisy input image.
        mu (float): The regularization parameter for the quadratic term.
        z (tf.Tensor): An initial estimate of the denoised image. Defaults to None.
        alpha (float): The regularization parameter for the TV term. Defaults to 0.01.
        n_iter (int): The number of iterations to run the algorithm. Defaults to 5.

    Returns:
        tf.Tensor: The denoised image.
    """
    device = x.device  # 获取输入x的设备

    if z is None:
        z = x

    def soft_threshold(x, threshold):
        return tf.math.sign(x) * tf.math.maximum(tf.math.abs(x) - threshold, tf.zeros_like(x))

    A = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tf.float32)
    B = tf.constant([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=tf.float32)
    A = tf.expand_dims(tf.expand_dims(A, axis=0), axis=0)
    B = tf.expand_dims(tf.expand_dims(B, axis=0), axis=0)
    AtA = tf.nn.conv2d(A, A, strides=[1, 1, 1, 1], padding='SAME')
    AtB = tf.nn.conv2d(A, B, strides=[1, 1, 1, 1], padding='SAME')

    for i in range(n_iter):
        # update x
        if c==1:
            x = tf.tile(x,[1,3,1,1])
        print("x.shape,AtB.shape",x.shape,AtB.shape)
        print("x.shape,AtB.shape",tf.nn.conv2d(x, AtB, strides=[1, 1, 1, 1], padding='SAME',data_format='NCHW'))
        x = (mu * z + tf.nn.conv2d(x, AtB, strides=[1, 1, 1, 1], padding='SAME',data_format='NCHW')) \
            / (mu + tf.nn.conv2d(tf.ones_like(x), AtA, strides=[1, 1, 1, 1], padding='SAME',data_format='NCHW'))
        if c==1:
            x = tf.reduce_mean(x,axis=1,keepdims=True)
        # update z
        d = x - z
        z = soft_threshold(z + alpha * d, alpha)

    return x
