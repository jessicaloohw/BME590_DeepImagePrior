def get_training_image(filename):
    """
    (DAVID) Load image for training (includes any pre-processing etc.)

    :param filename: the filename of the image
                     string
    :return: image: the training image [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                    float32
    """

    image = None
    return image


def calculate_metrics(image, metrics_name):
    """
    (DAVID) To calculate metrics for image quality evaluation

    :param image: [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
                  float32
    :return: metric: metric value (scalar)
                     float32
    """

    # Create different metrics here:
    if metrics_name == 'snr':
        snr = None
        return snr

    elif metrics_name == 'ssim':
        ssim = None
        return ssim