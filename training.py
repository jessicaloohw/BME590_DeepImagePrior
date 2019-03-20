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

    # Build network:
    x = tf.placeholder(tf.float32, shape=[1, None, None, 1])
    y = network.inference(x, network_name=NETWORK_NAME)
    loss = network.loss(y, x, loss_name=LOSS_NAME)

    # (KRISTEN) Create different optimizers here:
    if OPTIMIZER_TYPE == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Start session:
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(NUM_ITERATIONS):
            if i == 0:
                input_image = hf.get_training_image(FILENAME)
            _, input_image = sess.run([train_op, y], feed_dict={x: input_image})

            if i % ITERATIONS_TO_SAVE == 0:
                # Save model
                # Save image
                # Save metrics etc.
                pass


if __name__ == '__main__':
    main()