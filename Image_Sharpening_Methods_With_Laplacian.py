import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


"""

The 2 things to remember here is:
    1.  Normalize the input image before convert it to the frequency domain 
    2.  Normalize (Rescale) the result (after filtering) to [-1, 1]
Why?
    To solve the problem of too big values after enhancement
"""


def frequency_domain_laplacian_filter(file_path):
    """
    f(x) is an image in the spatial domain
    F(u) is an image on the frequency domain (the Fourier Transform of f(x))
    H(x) is the laplacian filter in the frequency domain
    g(x) is the enhanced image in the spatial domain
    P, Q are the number of rows and columns in the image
    c is a constant
    # Problem: we get too big values after add the filterd image to the original image,
     so we need to normalize input image before convert it to the frequency domain, as I did.
     This is the reason for normalization and using np.clip().

    :param file_path: The path to the image file
    :return: None
    """
    # Normalize the input image before convert it to the frequency domain to solve the problem of too big values into [0, 1] range
    f = cv.imread(file_path, 0)
    f = f / 255
    # 2dim FFT and shift the low frequency into the center of the image
    F = np.fft.fftshift(np.fft.fft2(f))

    # Laplacian filter in the frequency domain
    P, Q = F.shape
    H = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            H[u,v] = -4 * np.pi**2 * ((u-P/2)**2 + (v-Q/2)**2)

    # Multiplication of the Fourier Transform of the image and the filter
    laplacian_image = F * H
    # Inverse Fourier Transform of the multiplication result
    laplacian_image_IFFT = np.fft.ifft2(laplacian_image)
    # Retrieve the real part of the result to get the image.
    laplacian_image_IFFT_real = np.real(laplacian_image_IFFT)
    # Normalize (Rescale) the result to [-1, 1] to solve the problem of too big values
    old_range = np.max(laplacian_image_IFFT_real) - np.min(laplacian_image_IFFT_real)
    new_range = 1 - (-1)
    laplacian_image_IFFT_real_normalized = (((laplacian_image_IFFT_real - np.min(laplacian_image_IFFT_real)) * new_range) / old_range) - 1

    # Add the filtered image to the original image to get the image enhancement
    c = -1
    g = f + c * laplacian_image_IFFT_real_normalized
    # Normalize (Rescale) the result to [0, 1] to solve the problem of too big values
    g = np.clip(g, 0, 1)

    # Display the original image and the enhanced image side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(f, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(g, cmap='gray')
    axes[1].set_title('Image sharpened in Frequency Domain')
    plt.show()

    # Display it in the OpenCV way
    cv.imshow('Original', f)
    cv.imshow('Image sharpened in Frequency Domain', g)
    cv.waitKey(0)
    cv.destroyAllWindows()


def spatial_domain_laplacian_filter(file_path):
    """
    Applies a Laplacian operator to the grayscale image and stores the output image
    Display the original and the enhanced images together
    :param file_path: The path to the image file
    :return: None
    """

    # Loads an image
    if file_path is None:
        raise ValueError('file_path must be provided')
    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        raise Exception(f'Error reading image {file_path}')

    # Remove noise by applying a Gaussian blur
    image = cv.GaussianBlur(image, (3, 3), 0)

    # Apply Laplace function, convert back to uint8
    kernel_size = 3
    ddepth = cv.CV_16S   # ddepth is set to 16 to support negative values and avoid overflow
    dst = cv.Laplacian(image, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)

    # Enhance sharpness using the original image and the Laplacian result
    enhanced = cv.addWeighted(image, 1.5, abs_dst, -0.5, 0)

    # Show the original and the enhanced images together in plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Image Sharpened In Spatial Domain')
    plt.show()

    # Show the original and the enhanced images together
    cv.namedWindow("Original", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Image Sharpened In Spatial Domain", cv.WINDOW_AUTOSIZE)
    cv.imshow('Original', image)
    cv.imshow('Image Sharpened In Spatial Domain', enhanced)
    cv.waitKey(0)
    cv.destroyAllWindows()


def spatial_domain_laplacian_filter_with_kernels(file_path):
    """
    cv.filter2D explanation:
        Convolves an image with the kernel.
        The function applies an arbitrary linear filter to an image.
        In-place operation is supported.
        When the aperture is partially outside the image, the function interpolates outlier pixel values according to the specified border mode.
        The function does actually compute correlation, not the convolution
        That is, the kernel is not mirrored around the anchor point.
        If you need a real convolution, flip the kernel using flip and set the new anchor to (kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1).
    :param file_path:
    :return:
    """
    # Load your image using cv2.imread()
    input_image = cv.imread(file_path)  # Replace with your image path
    input_image = input_image.astype(np.float32) / 255
    # Define 4 laplacian filter kernels from the Digital Image Processing Book (Gonzalez, Woods, page 179)
    laplacian_kernel1 = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    laplacian_kernel2 = np.array([[1, 1, 1],
                                  [1, -8, 1],
                                  [1, 1, 1]])
    laplacian_kernel3 = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
    laplacian_kernel4 = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])

    # Apply the laplacian filter to the input image using cv2.filter2D()
    image_sharp1 = cv.filter2D(src=input_image, ddepth=-1, kernel=laplacian_kernel1)
    image_sharp2 = cv.filter2D(src=input_image, ddepth=-1, kernel=laplacian_kernel2)
    image_sharp3 = cv.filter2D(src=input_image, ddepth=-1, kernel=laplacian_kernel3)
    image_sharp4 = cv.filter2D(src=input_image, ddepth=-1, kernel=laplacian_kernel4)

    # Enhance sharpness using the original image and the Laplacian result
    image_sharp1 = input_image - image_sharp1
    image_sharp2 = input_image - image_sharp2
    image_sharp3 = input_image + image_sharp3
    image_sharp4 = input_image + image_sharp4
    # Normalize (Rescale) the result to [0, 1] to solve the problem of too big values
    image_sharp1[image_sharp1 < 0.0] = 0.0
    image_sharp1[image_sharp1 > 1.0] = 1.0
    image_sharp2[image_sharp2 < 0.0] = 0.0
    image_sharp2[image_sharp2 > 1.0] = 1.0
    image_sharp3[image_sharp3 < 0.0] = 0.0
    image_sharp3[image_sharp3 > 1.0] = 1.0
    image_sharp4[image_sharp4 < 0.0] = 0.0
    image_sharp4[image_sharp4 > 1.0] = 1.0

    # Display the original image and the 4 enhanced images side by side with plt
    fig, axes = plt.subplots(1, 6, figsize=(10, 7))
    axes[0].imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[1].imshow(cv.cvtColor(image_sharp1, cv.COLOR_BGR2RGB))
    axes[1].set_title('Sharpened with kernel1 (Spatial Domain)')
    axes[2].imshow(cv.cvtColor(image_sharp2, cv.COLOR_BGR2RGB))
    axes[2].set_title('Sharpened with kernel2 (Spatial Domain)')
    axes[3].imshow(cv.cvtColor(image_sharp3, cv.COLOR_BGR2RGB))
    axes[3].set_title('Sharpened with kernel3 (Spatial Domain)')
    axes[4].imshow(cv.cvtColor(image_sharp4, cv.COLOR_BGR2RGB))
    axes[4].set_title('Sharpened with kernel4 (Spatial Domain)')
    plt.show()

    cv.imshow('Original', input_image)
    cv.imshow('Sharpened with kernel1 (Spatial Domain)', image_sharp1)
    cv.imshow('Sharpened with kernel2 (Spatial Domain)', image_sharp2)
    cv.imshow('Sharpened with kernel3 (Spatial Domain)', image_sharp3)
    cv.imshow('Sharpened with kernel4 (Spatial Domain)', image_sharp4)

    cv.waitKey()
    cv.destroyAllWindows()


def main():
    print('start...')
    input_file_path ='C:\Idan\OpenU\image_processing\Code\moon_from_book.jpg' # input('Enter the gray image file path: ')

    frequency_domain_laplacian_filter(file_path=input_file_path)
    spatial_domain_laplacian_filter(file_path=input_file_path)
    spatial_domain_laplacian_filter_with_kernels(file_path=input_file_path)

    print('Done')


if __name__ == "__main__":
    main()


def from_someone_on_the_internet_nice_implementation():
    import cv2
    import numpy as np

    class ImageSharpening:
        def __init__(self):
            self.sharpeningKernel = np.zeros((3, 3), np.float32)
            self.sharpeningKernel[0, 1] = 1.0
            self.sharpeningKernel[1, 0] = 1.0
            self.sharpeningKernel[1, 1] = -4.0
            self.sharpeningKernel[1, 2] = 1.0
            self.sharpeningKernel[2, 1] = 1.0

        def sharpen_image(self, img):
            imgfloat = img.astype(np.float32) / 255
            imgLaplacian = cv2.filter2D(imgfloat, cv2.CV_32F, self.sharpeningKernel)
            res = imgfloat - imgLaplacian
            res[res < 0.0] = 0.0
            res[res > 1.0] = 1.0

            res = (res * 255).astype(np.uint8)
            return res

    # Load your image using cv2.imread()
    input_image = cv2.imread("C:\Idan\OpenU\image_processing\Code\my_moon.jpeg")  # Replace with your image path
    image_sharpening = ImageSharpening()
    sharpened_image = image_sharpening.sharpen_image(input_image)

    # Display the original image
    cv2.imshow("Original Image", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display or save the sharpened image
    cv2.imshow("Sharpened Image", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()