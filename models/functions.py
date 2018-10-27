import tensorflow as tf


# def leaky_relu(inputs, alpha=.1):
#     with tf.name_scope('leaky_relu') as name:
#         data = tf.identity(inputs, name='data')
#         return tf.maximum(data, alpha * data, name=name)

def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    """ LeakyReLU.
    Modified version of ReLU, introducing a nonzero gradient for negative
    input.
    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        alpha: `float`. slope.
        name: A name for this activation op (optional).
    Returns:
        A `Tensor` with the same type as `x`.
    References:
        Rectifier Nonlinearities Improve Neural Network Acoustic Models,
        Maas et al. (2013).
    Links:
        [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf]
        (http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """

    with tf.name_scope(name) as scope:
        m_x = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        x -= alpha * m_x

    x.scope = scope

    return x