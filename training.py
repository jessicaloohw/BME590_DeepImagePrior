import tensorflow as tf
import network
import helper_functions as hf

import os
from matplotlib.pyplot import imsave


def main():

    # Input parameters:
    IMAGE_NAME = '1_Raw Image'

    # Training parameters:
    NETWORK_NAME = 'unet'
    LOSS_NAME = 'mse'
    NUM_ITERATIONS = 1000
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.0001

    # Saving parameters:
    ITERATIONS_TO_SAVE = 100

    # Image:
    FILENAME = os.path.join('./Raw', '{}.tif'.format(IMAGE_NAME))

    # Results:
    SAVE_FOLDER = os.path.join('./results', IMAGE_NAME)
    count = 0
    while os.path.exists(SAVE_FOLDER):
        count += 1
        SAVE_FOLDER = '{}({})'.format(SAVE_FOLDER, count)
    os.mkdir(SAVE_FOLDER)

    WRITE_FILENAME = os.path.join(SAVE_FOLDER, 'metrics.txt')
    with open(WRITE_FILENAME, 'a') as wf:
        wf.write('PARAMETERS\nNetwork: {}\nLoss: {}\nOptimizer: {}\nLearning rate: {}\nNumber of iterations: {}'.format(
            NETWORK_NAME, LOSS_NAME, OPTIMIZER_TYPE, LEARNING_RATE, NUM_ITERATIONS))
        wf.write('\n\nIteration\tSNR\tSSIM')

    # Placeholders:
    z = tf.placeholder(tf.float32, shape=[1, None, None, 32]) # input noise
    x = tf.placeholder(tf.float32, shape=[1, None, None, 1])  # input image

    # Network:
    y = network.inference(NETWORK_NAME, z)
    loss = network.loss(y, x, LOSS_NAME)

    # (KRISTEN) Create different optimizers here:
    if OPTIMIZER_TYPE == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    elif OPTIMIZER_TYPE == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Start session:
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        input_image = hf.get_training_image(FILENAME)
        input_noise = hf.get_noise_matrix(input_image.shape[1], input_image.shape[2], 32)

        for i in range(NUM_ITERATIONS):
            _, output_image = sess.run([train_op, y], feed_dict={z: input_noise,
                                                                 x: input_image
                                                                 }
                                       )

            if i % ITERATIONS_TO_SAVE == 0:
                # Save model (not really necessary, I think)

                # Save image
                save_filename = os.path.join(SAVE_FOLDER, 'iteration_{}.tif'.format(i))
                imsave(save_filename, output_image[0, :, :, 0])

                # Calculate metrics
                snr = hf.calculate_metrics(input_image, output_image, 'snr')
                ssim = hf.calculate_metrics(input_image, output_image, 'ssim')
                with open(WRITE_FILENAME, 'a') as wf:
                    wf.write('\n{}\t{}\t{}'.format(i, snr, ssim))

                # Display
                print('Iteration {}/{}\t| SNR: {}\tSSIM: {}'.format(i, NUM_ITERATIONS, snr, ssim))

    print('Completed.')


if __name__ == '__main__':
    main()
