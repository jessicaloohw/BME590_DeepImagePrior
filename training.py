import tensorflow as tf
import network
import helper_functions as hf
import numpy as np
import sys
import os
from matplotlib.pyplot import imsave


def main():

    ########################################## USER INPUT ##############################################################

    # Training parameters:
    if len(sys.argv) >= 8:
        IMAGE_NAME = sys.argv[1]            # IMAGE_NAME = '1'
        NETWORK_NAME = sys.argv[2]          # 'unet', 'deep_decoder'
        LOSS_NAME = sys.argv[3]             # 'mse', 'l1', 'mse_l1', 'mse_with_tv_reg', 'mse_with_edge_reg'
        OPTIMIZER_TYPE = sys.argv[4]        # 'sgd', 'adam'
        LEARNING_RATE = float(sys.argv[5])
        NUM_ITERATIONS = int(sys.argv[6])
        ITERATIONS_TO_SAVE = int(sys.argv[7])

        if len(sys.argv) == 11:
            w_h = float(sys.argv[8])
            w_v = float(sys.argv[9])
            w_mse = float(sys.argv[10])
        else:
            w_h = None
            w_v = None
            w_mse = None

    else:
        print('Not enough input parameters.')
        return

    ####################################################################################################################

    # Load images:
    RAW_FILENAME = os.path.join('./Raw', '{}_Raw Image.tif'.format(IMAGE_NAME))
    AVERAGED_FILENAME = os.path.join('./Averaged', '{}_Averaged Image.tif'.format(IMAGE_NAME))

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
    VALID_NETWORK_NAMES = ["unet", "deep_decoder"]
    VALID_OPTIMIZER_TYPES = ["sgd", "adam"]
    VALID_LOSS_NAMES = ["mse", "l1", "mse_l1", "mse_with_tv_reg", "mse_with_edge_reg"]

    if not (NETWORK_NAME in VALID_NETWORK_NAMES):
        print("Error: {} network does not exist.".format(NETWORK_NAME))
        return
    if not (OPTIMIZER_TYPE  in VALID_OPTIMIZER_TYPES):
        print("Error: {} optimizer does not exist.".format(OPTIMIZER_TYPE))
        return
    if not (LOSS_NAME in VALID_LOSS_NAMES):
        print("Error: {} loss does not exist.".format(LOSS_NAME))
        return

    # Create folder to save results:
    SAVE_FOLDER = os.path.join('./results', IMAGE_NAME)
    count = 0
    CHECK_FOLDER = SAVE_FOLDER
    while os.path.exists(CHECK_FOLDER):
        count += 1
        CHECK_FOLDER = '{}({})'.format(SAVE_FOLDER, count)
    SAVE_FOLDER = CHECK_FOLDER
    os.mkdir(SAVE_FOLDER)

    WRITE_FILENAME = os.path.join(SAVE_FOLDER, 'metrics.txt')
    with open(WRITE_FILENAME, 'a') as wf:
        wf.write('PARAMETERS\nNetwork: {}\nLoss: {}\nOptimizer: {}\nLearning rate: {}\nNumber of iterations: {}'.format(
            NETWORK_NAME, LOSS_NAME, OPTIMIZER_TYPE, LEARNING_RATE, NUM_ITERATIONS))
        wf.write('\n\nw_h: {}\nw_v: {}\nw_mse: {}'.format(w_h, w_v, w_mse))
        wf.write('\n\nIteration\tLoss\tSNR\tCNR\tSSIM')

    # Get input noise:
    if NETWORK_NAME == "unet":
        input_noise = hf.get_noise_matrix(input_image.shape[1], input_image.shape[2], 32)
    elif NETWORK_NAME == "deep_decoder":
        input_noise = hf.get_noise_matrix(input_image.shape[1]/(2**4), input_image.shape[2]/(2**4), 64)

    # Save inputs:
    save_filename = os.path.join(SAVE_FOLDER, 'input_image.tif')
    imsave(save_filename, input_image[0, :, :, 0], cmap='gray')

    save_filename = os.path.join(SAVE_FOLDER, 'ground_truth.tif')
    imsave(save_filename, ground_truth[0, :, :, 0], cmap='gray')

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
    if LOSS_NAME == "mse_with_edge_reg" or LOSS_NAME == "mse_with_tv_reg":
        loss, mse, edge_h, edge_v = network.loss(y, x, LOSS_NAME, w_h, w_v, w_mse)
    else:
        loss = network.loss(y, x, LOSS_NAME)

    # Update moving mean and variance for batch normalization (if required):
    if NETWORK_NAME == "deep_decoder":
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # (KRISTEN) Create different optimizers here:
    if OPTIMIZER_TYPE == "sgd":
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    elif OPTIMIZER_TYPE == "adam":
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
                if LOSS_NAME == "mse_with_edge_reg" or LOSS_NAME == "mse_with_tv_reg":
                    _, output_image, loss_i, mse_i, edge_h_i, edge_v_i = sess.run([train_op, y, loss, mse, edge_h, edge_v], feed_dict={z: input_noise,
                                                                                                                                       x: input_image})
                else:
                    _, output_image, loss_i = sess.run([train_op, y, loss], feed_dict={z: input_noise,
                                                                                       x: input_image})
            elif NETWORK_NAME == "deep_decoder":
                if LOSS_NAME == "mse_with_edge_reg" or LOSS_NAME == "mse_with_tv_reg":
                    _, _, output_image, loss_i, mse_i, edge_h_i, edge_v_i = sess.run([update_op, train_op, y, loss, mse, edge_h, edge_v], feed_dict={z: input_noise,
                                                                                                                                                     x: input_image})
                else:
                    _, _, output_image, loss_i = sess.run([update_op, train_op, y, loss], feed_dict={z: input_noise,
                                                                                                     x: input_image})

            if i % ITERATIONS_TO_SAVE == 0:

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
                if LOSS_NAME == "mse_with_edge_reg" or LOSS_NAME == "mse_with_tv_reg":
                    print('Iteration {}/{}\t| Loss: {}\tSNR: {}\tCNR: {}\tSSIM: {}\tMSE: {}\tEdge_h: {}\tEdge_v: {}'.format(i, NUM_ITERATIONS, loss_i, snr_i,
                                                                                                                            cnr_i, ssim_i, mse_i, edge_h_i, edge_v_i))
                else:
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
