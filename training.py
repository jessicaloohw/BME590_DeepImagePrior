import tensorflow as tf
import network
import helper_functions as hf


def main():

    # Input parameters:
    FILENAME = ''

    # Training parameters:
    NETWORK_NAME = 'unet'
    LOSS_NAME = 'mse'
    NUM_ITERATIONS = 1000
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.0001

    # Saving parameters:
    ITERATIONS_TO_SAVE = 100
    SAVE_FOLDER = ''

    # Placeholders:
    z = tf.placeholder(tf.float32, shape=[1, None, None, 1]) # input noise
    x = tf.placeholder(tf.float32, shape=[1, None, None, 1]) # input image
    c = tf.placeholder(tf.int32)                             # number of channels

    # Network:
    y = network.inference(NETWORK_NAME, z, c)
    loss = network.loss(y, x, LOSS_NAME)

    # (KRISTEN) Create different optimizers here:
    if OPTIMIZER_TYPE == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    if OPTIMIZER_TYPE == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Start session:
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        input_image = hf.get_training_image(FILENAME)
        input_noise = hf.get_noise_matrix(input_image.shape[0], input_image.shape[1], 32)

        for i in range(NUM_ITERATIONS):
            _, output_image = sess.run([train_op, y], feed_dict={z: input_noise,
                                                                 x: input_image,
                                                                 c: input_image.shape[2]
                                                                 }
                                       )

            if i % ITERATIONS_TO_SAVE == 0:
                # Save model
                # Save image
                # Save metrics etc.
                pass


if __name__ == '__main__':
    main()
