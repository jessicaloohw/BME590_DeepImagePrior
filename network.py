import tensorflow as tf

def inference(x, network_name):
    """
    (JESSICA) Network architecture

    :param x: the input image
              float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    :param network_name: to select architecture
                         string
    :return: output: the denoised image
                     float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    """

    # Create different architectures here:
    if network_name == 'unet':
        output = None
        return output


def loss(y, x, loss_name):
    """
    (KRISTEN) Loss function

    :param y: the predicted output
              float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    :param x: the "ground truth"
              float32 tensor [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    :param loss_name: to select loss
                      string
    :return: loss: loss value (scalar)
                   float32
    """

    # Create different losses here:
    if loss_name == 'mse':
        loss = None
        return loss