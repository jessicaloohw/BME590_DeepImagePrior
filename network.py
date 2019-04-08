import tensorflow as tf
import numpy as np

def inference(network_name, z, height, width, channels):
    """
    (JESSICA) Network architecture

    :param network_name: to select architecture
                         string
    :param z: the input noise matrix
              float32 [1, NOISE_HEIGHT, NOISE_WIDTH, NOISE_CHANNELS]
    :param height: NOISE_HEIGHT
                   int
    :param width: NOISE_WIDTH
                  int
    :param channels: NOISE_CHANNELS
                     int
    :return: output: the denoised image
                     float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    """

    initializer = tf.contrib.layers.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    # Input noise:
    print(z)

    # Create different architectures here:
    if network_name == 'unet':

        with tf.variable_scope('encoder1'):
            x2 = tf.layers.conv2d(z, 64, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x2 = tf.layers.conv2d(x2, 64, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x2 = tf.nn.relu(x2, name='relu')
            print(x2)

        with tf.variable_scope('encoder2'):
            x3 = tf.layers.max_pooling2d(x2, [2, 2], [2, 2], "VALID", name='pool')
            x3 = tf.layers.conv2d(x3, 128, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x3 = tf.layers.conv2d(x3, 128, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x3 = tf.nn.relu(x3, name='relu')
            print(x3)

        with tf.variable_scope('encoder3'):
            x4 = tf.layers.max_pooling2d(x3, [2, 2], [2, 2], "VALID", name='pool')
            x4 = tf.layers.conv2d(x4, 256, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x4 = tf.layers.conv2d(x4, 256, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x4 = tf.nn.relu(x4, name='relu')
            print(x4)

        with tf.variable_scope('encoder4'):
            x5 = tf.layers.max_pooling2d(x4, [2, 2], [2, 2], "VALID", name='pool')
            x5 = tf.layers.conv2d(x5, 512, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x5 = tf.layers.conv2d(x5, 512, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x5 = tf.nn.relu(x5, name='relu')
            print(x5)

        with tf.variable_scope('bottleneck'):
            x6 = tf.layers.max_pooling2d(x5, [2, 2], [2, 2], "VALID", name='pool')
            x6 = tf.layers.conv2d(x6, 1024, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x6 = tf.layers.conv2d(x6, 1024, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x6 = tf.nn.relu(x6, name='relu')

        with tf.variable_scope('skip4'):
            x6 = tf.layers.conv2d_transpose(x6, 512, [2, 2], [2, 2], "SAME",
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=initializer,
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer,
                                            name='deconv')
            print(x6)
            x7 = tf.concat([x5, x6], axis=3, name='concat')  # encoder4 + skip4

        with tf.variable_scope('decoder4'):
            x7 = tf.layers.conv2d(x7, 512, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x7 = tf.layers.conv2d(x7, 512, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x7 = tf.nn.relu(x7, name='relu')

        with tf.variable_scope('skip3'):
            x7 = tf.layers.conv2d_transpose(x7, 256, [2, 2], [2, 2], "SAME",
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=initializer,
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer,
                                            name='deconv')

            print(x7)
            x8 = tf.concat([x4, x7], axis=3, name='concat')  # encoder3 + skip3

        with tf.variable_scope('decoder3'):
            x8 = tf.layers.conv2d(x8, 256, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x8 = tf.layers.conv2d(x8, 256, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x8 = tf.nn.relu(x8, name='relu')

        with tf.variable_scope('skip2'):
            x8 = tf.layers.conv2d_transpose(x8, 128, [2, 2], [2, 2], "SAME",
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=initializer,
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer,
                                            name='deconv')

            print(x8)
            x9 = tf.concat([x3, x8], axis=3, name='concat')  # encoder2 + skip2

        with tf.variable_scope('decoder2'):
            x9 = tf.layers.conv2d(x9, 128, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv1')
            x9 = tf.layers.conv2d(x9, 128, [3, 3], [1, 1], "SAME",
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  name='conv2')
            x9 = tf.nn.relu(x9, name='relu')

        with tf.variable_scope('skip1'):
            x9 = tf.layers.conv2d_transpose(x9, 64, [2, 2], [2, 2], "SAME",
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=initializer,
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer,
                                            name='deconv')
            print(x9)
            x10 = tf.concat([x2, x9], axis=3, name='concat')  # encoder1 + skip1

        with tf.variable_scope('decoder1'):
            x10 = tf.layers.conv2d(x10, 64, [3, 3], [1, 1], "SAME",
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   name='conv1')
            x10 = tf.layers.conv2d(x10, 64, [3, 3], [1, 1], "SAME",
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer,
                                   name='conv2')
            x10 = tf.nn.relu(x10, name='relu')
            print(x10)

        with tf.variable_scope('prediction'):
            output = tf.layers.conv2d(x10, 1, [1, 1], [1, 1], "SAME",
                                      activation=tf.nn.sigmoid,
                                      use_bias=True,
                                      kernel_initializer=initializer,
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_regularizer=regularizer,
                                      bias_regularizer=regularizer,
                                      name='conv')
            print(output)

    elif network_name == "deep_decoder":

        # conv --> bilinear upsampling (if required) --> ReLU  --> normalization

        with tf.variable_scope('decoder1'):
            x = tf.layers.conv2d(z, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.image.resize_bilinear(images=x,
                                         size=tf.convert_to_tensor([int(height*(2**1)), int(width*(2**1))]),
                                         name='upsample')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('decoder2'):
            x = tf.layers.conv2d(x, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.image.resize_bilinear(images=x,
                                         size=tf.convert_to_tensor([int(height*(2**2)), int(width*(2**2))]),
                                         name='upsample')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('decoder3'):
            x = tf.layers.conv2d(x, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.image.resize_bilinear(images=x,
                                         size=tf.convert_to_tensor([int(height*(2**3)), int(width*(2**3))]),
                                         name='upsample')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('decoder4'):
            x = tf.layers.conv2d(x, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.image.resize_bilinear(images=x,
                                         size=tf.convert_to_tensor([int(height*(2**4)), int(width*(2**4))]),
                                         name='upsample')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('decoder5'):
            x = tf.layers.conv2d(x, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('decoder6'):
            x = tf.layers.conv2d(x, channels, [1, 1], [1, 1], "SAME",
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=initializer,
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 name='conv')
            x = tf.nn.relu(x, name="relu")
            x = tf.layers.batch_normalization(x, training=True, name='bn')

            print(x)

        with tf.variable_scope('prediction'):
            output = tf.layers.conv2d(x, 1, [1, 1], [1, 1], "SAME",
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=initializer,
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_regularizer=regularizer,
                                      bias_regularizer=regularizer,
                                      name='conv')

            print(output)

    return output


def loss(y, x, loss_name):
    """
    (KRISTEN) Loss function

    :param y: the predicted output
              float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    :param x: the "ground truth"
              float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    :param loss_name: to select loss
                      string
    :return: loss: loss value (scalar)
                   float32
    """

    # Create different losses here:
    if loss_name == 'mse':
        main_loss = tf.reduce_mean((y-x)**2)
    if loss_name == 'l1':
        main_loss = tf.reduce_mean(np.abs(y-x))
    if loss_name == 'mse_l1':
#https://iopscience.iop.org/article/10.1088/1612-202X/aaaeb0/meta
        reg = 0.1
        main_loss = tf.reduce_sum(np.mean((y-x)**2)+reg*np.abs(y-x))
    tf.losses.add_loss(main_loss)
    total_loss = tf.losses.get_total_loss()
    return total_loss
