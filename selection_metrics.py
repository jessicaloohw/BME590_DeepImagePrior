import os
from matplotlib.image import imread, imsave
import helper_functions as hf
# from tqdm import tqdm

def main():

    ####################################### USER-INPUT ##################################################

    FOLDER = ('/home/ubuntu/PycharmProjects/DeepImagePrior/BME590_DeepImagePrior'
              '/results/UNET_MSE_EDGE/1(1)')
    IMAGE_NAME = '1'
    START_ITER = 0
    END_ITER = 50000

    ######################################################################################################


    # Create save folders:
    SAVE_FOLDER = os.path.join(FOLDER, 'selection_metrics')
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    # Initialise write file:
    WRITE_FILENAME = os.path.join(SAVE_FOLDER, 'metrics.txt')
    with open(WRITE_FILENAME, 'a') as wf:
        wf.write('Iteration\tCNR\tSSIM')

    # Load original noisy image:
    noisy_filename = os.path.join(FOLDER, 'input_image.tif')
    noisy_image = hf.get_training_image(noisy_filename)

    # Load ground truth image:
    truth_filename = os.path.join(FOLDER, 'ground_truth.tif')
    truth_image = hf.get_training_image(truth_filename)

    # Calculate initial metrics:
    cnr = hf.calculate_metrics(truth_image, noisy_image, 'cnr', IMAGE_NAME)
    ssim = hf.calculate_metrics(truth_image, noisy_image, 'ssim', IMAGE_NAME)

    # Write to file:
    with open(WRITE_FILENAME, 'a') as wf:
        wf.write('\ninput_image\t{}\t{}'.format(cnr, ssim))

    # Keep track:
    track_iter = []
    track_cnr = []
    track_ssim = []

    print('Calculating metrics...')

    for i in range(START_ITER, END_ITER+1, 1000):

        # Load reconstructed image:
        denoised_filename = os.path.join(FOLDER, 'iteration_{}.tif'.format(i))
        denoised_image = hf.get_training_image(denoised_filename)

        # Calculate metrics:
        cnr = hf.calculate_metrics(truth_image, denoised_image, 'cnr', IMAGE_NAME)
        ssim = hf.calculate_metrics(truth_image, denoised_image, 'ssim', IMAGE_NAME)

        # Write to file:
        with open(WRITE_FILENAME, 'a') as wf:
            wf.write('\n{}\t{}\t{}'.format(i, cnr, ssim))

        # Track:
        track_iter.append(i)
        track_cnr.append(cnr)
        track_ssim.append(ssim)

    # Plot:
    plot_filename = os.path.join(SAVE_FOLDER, 'cnr.tif')
    hf.plot_metrics(x=track_iter,
                    y=track_cnr,
                    title='cnr',
                    save_filename=plot_filename,
                    calculate_max=True,
                    calculate_min=False)

    plot_filename = os.path.join(SAVE_FOLDER, 'ssim.tif')
    hf.plot_metrics(x=track_iter,
                    y=track_ssim,
                    title='ssim',
                    save_filename=plot_filename,
                    calculate_max=True,
                    calculate_min=False)

    print('Completed.')


if __name__ == "__main__":
    main()