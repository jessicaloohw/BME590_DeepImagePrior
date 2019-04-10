import tensorflow as tf
import network
import helper_functions as hf

import sys
import os
from matplotlib.pyplot import imsave


def main():

    ########################################## USER INPUT ##############################################################

    # System input parameters:
    IMAGE_NAME = sys.argv[1]
    # IMAGE_NAME = '1'

    # Training parameters:
    NETWORK_NAME = 'deep_decoder'   # 'unet', 'deep_decoder'
    LOSS_NAME = 'mse_l1'            # 'mse', 'l1', 'mse_l1'
    NUM_ITERATIONS = 1000
    OPTIMIZER_TYPE = 'adam'         # 'sgd', 'adam'
    LEARNING_RATE = 0.0001

    # Saving parameters:
    ITERATIONS_TO_SAVE = 100

    ####################################################################################################################

    # Load image:
    RAW_FILENAME = os.path.join('./Raw', '{}_Raw Image.tif'.format(IMAGE_NAME))
    AVERAGED_FILENAME = os.path.join('./Averaged', '{}_Averaged Image.tif'.format(IMAGE_NAME))

    # Get inputs:
    try:
        input_image = hf.get_training_image(RAW_FILENAME)
    except:
        print("Error loading {}".format(RAW_FILENAME))
        return

    try:
        ground_truth = hf.get_training_image(AVERAGED_FILENAME)
    except:
        print("Error loading {}".format(AVERAGED_FILENAME))
        return

    # Validate settings:
    if not (NETWORK_NAME == "unet" or NETWORK_NAME == "deep_decoder"):
        print("Error: {} network does not exist.".format(NETWORK_NAME))
        return
    if not (OPTIMIZER_TYPE == "sgd" or OPTIMIZER_TYPE == "adam"):
        print("Error: {} optimizer does not exist.".format(OPTIMIZER_TYPE))
        return
    if not (LOSS_NAME == "mse" or LOSS_NAME == "l1" or LOSS_NAME == "mse_l1"):
        print("Error: {} loss does not exist.".format(LOSS_NAME))
        return

    # Create folder to save results:
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
        wf.write('\n\nIteration\tLoss\tSNR\tCNR\tSSIM')

    if NETWORK_NAME == "unet":
        input_noise = hf.get_noise_matrix(input_image.shape[1], input_image.shape[2], 32)
    elif NETWORK_NAME == "deep_decoder":
        input_noise = hf.get_noise_matrix(input_image.shape[1]/(2**4), input_image.shape[2]/(2**4), 64)

    # Save inputs:
    save_filename = os.path.join(SAVE_FOLDER, 'input_image.tif')
    imsave(save_filename, input_image[0, :, :, 0], cmap='gray')

    # Calculate initial metrics:
    snr_i = hf.calculate_metrics(ground_truth, input_image, 'snr')
    cnr_i = hf.calculate_metrics(ground_truth, input_image, 'cnr')
    ssim_i = hf.calculate_metrics(ground_truth, input_image, 'ssim')
    with open(WRITE_FILENAME, 'a') as wf:
        wf.write('\ninput_image\tN/A\t{}\t{}\t{}'.format(snr_i, cnr_i, ssim_i))

    # Placeholders:
    z = tf.placeholder(tf.float32, shape=[1, None, None, input_noise.shape[3]]) # input noise
    x = tf.placeholder(tf.float32, shape=[1, None, None, 1])                    # input image

    # Network:
    y = network.inference(NETWORK_NAME, z,
                          height=input_noise.shape[1],
                          width=input_noise.shape[2],
                          channels=input_noise.shape[3])
    loss = network.loss(y, x, LOSS_NAME)

    # Update moving mean and variance for batch normalization (if required):
    if NETWORK_NAME == "deep_decoder":
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # (KRISTEN) Create different optimizers here:
    if OPTIMIZER_TYPE == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    elif OPTIMIZER_TYPE == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Start session:
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Keep track of metrics:
        track_iter = []
        track_loss = []
        track_snr = []
        track_cnr = []
        track_ssim = []

        for i in range(NUM_ITERATIONS+1):

            if NETWORK_NAME == "unet":
                _, output_image, loss_i = sess.run([train_op, y, loss], feed_dict={z: input_noise,
                                                                                   x: input_image})
            elif NETWORK_NAME == "deep_decoder":
                _, _, output_image, loss_i = sess.run([update_op, train_op, y, loss], feed_dict={z: input_noise,
                                                                                                 x: input_image})

            if i % ITERATIONS_TO_SAVE == 0:
                # Save model (not really necessary, I think)

                # Save image:
                save_filename = os.path.join(SAVE_FOLDER, 'iteration_{}.tif'.format(i))
                imsave(save_filename, output_image[0, :, :, 0], cmap='gray')

                # Calculate metrics:
                snr_i = hf.calculate_metrics(ground_truth, output_image, 'snr')
                cnr_i = hf.calculate_metrics(ground_truth, output_image, 'cnr')
                ssim_i = hf.calculate_metrics(ground_truth, output_image, 'ssim')
                with open(WRITE_FILENAME, 'a') as wf:
                    wf.write('\n{}\t{}\t{}\t{}\t{}'.format(i, loss_i, snr_i, cnr_i, ssim_i))

                # Display:
                print('Iteration {}/{}\t| Loss: {}\tSNR: {}\tCNR: {}\tSSIM: {}'.format(i, NUM_ITERATIONS, loss_i, snr_i, cnr_i, ssim_i))

                # Track:
                track_iter.append(i)
                track_loss.append(loss_i)
                track_snr.append(snr_i)
                track_cnr.append(cnr_i)
                track_ssim.append(ssim_i)

        # Plot:
        hf.plot_metrics(track_iter, track_loss, 'loss', os.path.join(SAVE_FOLDER, 'loss.tif'))
        hf.plot_metrics(track_iter, track_snr, 'snr', os.path.join(SAVE_FOLDER, 'snr.tif'))
        hf.plot_metrics(track_iter, track_cnr, 'cnr', os.path.join(SAVE_FOLDER, 'cnr.tif'))
        hf.plot_metrics(track_iter, track_ssim, 'ssim', os.path.join(SAVE_FOLDER, 'ssim.tif'))

    print('Completed.')


if __name__ == '__main__':
    main()
