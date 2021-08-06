import tensorflow as tf

def _rbf_kernel(X, Y, sigma=1., wt=1., K_XY_only=False):

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    XYsqnorm = tf.maximum(-2 * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XY = wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = tf.maximum(-2 * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XX = wt * tf.exp(-gamma * XXsqnorm)
    K_YY = wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, wt