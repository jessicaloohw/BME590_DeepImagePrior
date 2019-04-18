import os
from matplotlib.image import imread, imsave
import helper_functions as hf

def main():

    ######################################### USER-INPUT ###############################################

    FOLDER = ('/home/ubuntu/PycharmProjects/DeepImagePrior/BME590_DeepImagePrior/results'
              '/unet_mse_edge/5')
    ITER_NUM = 31000

    ####################################################################################################

    read_filename = os.path.join(FOLDER, 'iteration_{}.tif'.format(ITER_NUM))
    image = imread(read_filename)
    image = image[:, :, 0]

    save_filename = os.path.join(FOLDER, 'iteration_{}_enhanced.tif'.format(ITER_NUM))
    enhanced = hf.imadjust(image)
    imsave(save_filename, enhanced, cmap='gray')

if __name__ == '__main__':
    main()