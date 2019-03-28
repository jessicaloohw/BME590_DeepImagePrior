import helper_functions as hf
import numpy as np

# just for testing helper functions outside of tensorflow
def main():
    print("hello")
    image = hf.get_training_image('Raw/3_Raw Image.tif')
    noise = hf.get_noise_matrix(image.shape[1], image.shape[2], 1)
    final = np.add(image, noise)
    metric = hf.calculate_metrics(final, image, "snr")

if __name__ == '__main__':

    main()