import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import rank


def main():
    # Load the original image
    img = cv.imread(r'C:\Idan\OpenU\image_processing\Code\embedded_squares.JPG', 0)

    # Create a square shape kernel/footprint/structured element, according to the squares in the image
    footprint = np.ones((15, 15))

    # Local equalization
    # rank.equalize() - Equalize image using local histogramming (based on the given structuring element).
    # footprint – The neighborhood expressed as an ndarray of 1's and 0's.
    img_local = rank.equalize(img, footprint=footprint)

    # Plot the original image and the local equalized one
    plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_local, cmap='gray'), plt.title('Local equalization')
    plt.xticks([]), plt.yticks([])
    # Plot the Histograms of the original image and the local equalized one
    plt.subplot(223), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Original Hist')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.hist(img_local.ravel(), 256, [0, 256]), plt.title('Local equalization Hist')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()
