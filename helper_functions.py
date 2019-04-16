import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as calc_ssim
import matplotlib.widgets as widgets
from PIL import Image
import bisect
from numba import jit


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

def calculate_metrics(input_image, output_image, metrics_name, image_number):
    """
    (DAVID) To calculate metrics for image quality evaluation

    :param input_image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                        float32 ground truth
    :param output_image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                        float32 prediction
    :param image_number: which image is it
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
        if image_number == 10:
            feature1 = output_image[120:160,105:145]
            feature2 = output_image[185:205,72:132]
            feature3 = output_image[125:155,710:740]
            feature4 = output_image[215:250,175:215]
        elif image_number == 11:
            feature1 = output_image[169:209, 216:256]
            feature2 = output_image[248:268, 205:265]
            feature3 = output_image[138:168, 730:760]
            feature4 = output_image[281:316, 53:93]
        elif image_number == 12:
            feature1 = output_image[191:231, 303:343]
            feature2 = output_image[260:280, 285:345]
            feature3 = output_image[182:212, 763:793]
            feature4 = output_image[258:293, 815:855]
        elif image_number == 13:
            feature1 = output_image[187:227, 303:343]
            feature2 = output_image[255:275, 298:358]
            feature3 = output_image[150:180, 699:729]
            feature4 = output_image[276:311, 492:532]
        elif image_number == 14:
            feature1 = output_image[26:66, 265:305]
            feature2 = output_image[96:116, 115:175]
            feature3 = output_image[37:67, 667:697]
            feature4 = output_image[115:150, 120:160]
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

@jit
def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # From:
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src, bins=list(range(256)), range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst
