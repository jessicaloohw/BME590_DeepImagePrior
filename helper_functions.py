import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as calc_ssim
import matplotlib.widgets as widgets
from PIL import Image

def get_training_image(filename):
    """
    (DAVID) Load image for training (includes any pre-processing etc.)

    :param filename: the filename of the image
                     string
    :return: image: the training image [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                    float32
    """

    image = mpimg.imread(filename)
    # hard coded to crop to smallest image 496x928
    image = image[0:496, 0:928]
    image = np.float32(image)
    # normalize
    image = image/255
    # creating channels and changes shape to expected output
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    """
    # debugging
    print(image.shape)
    plt.imshow(image[0, :, :, 0])
    plt.show()
    """
    return image


def get_noise_matrix(h, w, c):
    """
    (DAVID) Create random uniform noise matrix

    :param h: height
    :param w: width
    :param c: number of channels
    :return: noise [1, h, w, c]
    """

    noise = np.random.rand(h, w, c)
    noise = np.expand_dims(noise, axis=0)

    """
    # debugging
    plt.imshow(noise[0, :, :, 0])
    plt.show()
    """
    return noise

def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    #print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    #print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    #print('used button  : ', eclick.button)
    global pos
    pos = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata),
           int(erelease.ydata)]

def calculate_metrics(input_image, output_image, metrics_name):
    """
    (DAVID) To calculate metrics for image quality evaluation

    :param input_image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                  float32 noisy image
    :param output_image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
        float32 denoised image
    :param metrics_name: snr or ssim
    :return metric: metric value (scalar)
                     float32
    """

    # assuming single channel since image is single channel
    input_image = input_image[0, :, :, 0]
    output_image = output_image[0, :, :, 0]

    # Create different metrics here:
    # using predetermined patch of background for noise
    # peak SNR
    if metrics_name == 'snr':
        output_noise = output_image[319:391, 836:908]
        snr = output_image.max()/np.var(output_noise)
        """
        # debugging
        plt.imshow(input_noise)
        plt.show()
        print(snr)
        """
        return snr

    elif metrics_name == 'ssim':
        input_image = input_image/np.amax(input_image)
        ssim = calc_ssim(input_image, output_image,
                         data_range=output_image.max()-output_image.min())
        """
        # debugging
                print(np.amax(output_image))
        print(np.amax(input_image))
        print(input_image.shape)
        print(output_image.shape)
        plt.imshow(input_image)
        plt.show()
        print(ssim)
        """

        return ssim

    # displays image. select foreground
    elif metrics_name == 'cnr':
        plt.imshow(output_image)
        ax = plt.gca()
        rs = widgets.RectangleSelector(ax, onselect, drawtype='box',
                                       rectprops=dict(facecolor='red',
                                                      edgecolor='black',
                                                      alpha=0.5, fill=False))
        plt.show()
        """ debugging print(pos) """

        feature = output_image[pos[1]:pos[3], pos[0]:pos[2]]
        output_noise = output_image[319:391, 836:908]
        cnr = np.abs(np.mean(feature)-np.mean(output_noise))/np.sqrt(0.5*(
            np.var(feature)+np.var(output_noise)))

        return cnr
