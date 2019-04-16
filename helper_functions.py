import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as calc_ssim
import matplotlib.widgets as widgets
from PIL import Image
import bisect


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
                        float32 ground truth
    :param output_image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                        float32 prediction
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
        snr = output_image.max()/(np.var(output_noise) + 1e-8)
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

    # hard coded for image 1
    # average CNR of 4 selected regions
    elif metrics_name == 'cnr':
        """ interactive
        plt.imshow(output_image)
        ax = plt.gca()
        rs = widgets.RectangleSelector(ax, onselect, drawtype='box',
                                       rectprops=dict(facecolor='red',
                                                      edgecolor='black',
                                                      alpha=0.5, fill=False))
        plt.show()
        # print(pos)

        feature = output_image[pos[1]:pos[3], pos[0]:pos[2]]
        """
        output_noise = output_image[319:391, 836:908]
        feature1 = output_image[51:67,197:269]
        feature2 = output_image[38:67,568:643]
        feature3 = output_image[115:140,158:241]
        feature4 = output_image[131:165,763:809]
        cnr1 = np.abs(np.mean(feature1)-np.mean(output_noise))/(np.sqrt(0.5*(
            np.var(feature1)+np.var(output_noise))) + 1e-8)
        cnr2 = np.abs(np.mean(feature2)-np.mean(output_noise))/(np.sqrt(0.5*(
            np.var(feature2)+np.var(output_noise))) + 1e-8)
        cnr3 = np.abs(np.mean(feature3)-np.mean(output_noise))/(np.sqrt(0.5*(
            np.var(feature3)+np.var(output_noise))) + 1e-8)
        cnr4 = np.abs(np.mean(feature4)-np.mean(output_noise))/(np.sqrt(0.5*(
            np.var(feature4)+np.var(output_noise))) + 1e-8)
        cnr = np.mean([cnr1,cnr2,cnr3,cnr4])

        return cnr

def plot_metrics(x, y, title='', save_filename=None):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('iterations')
    if not (save_filename is None):
        plt.savefig(save_filename)


def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python/44529776#44529776
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst