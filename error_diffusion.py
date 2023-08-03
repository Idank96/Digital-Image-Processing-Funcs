import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import cv2


def find_closest_value(value, m):
    '''
    Find the closest value to the current pixel
    Explanation:
    1.  Find the range between levels by dividing 255 by the number of levels - 1
    2.  Find the level of the current pixel by dividing the value by the range between levels
        For example if the range is 63.75 and the value is 127, then the division is 1.99215 so rounding it to the
        nearest integer will give us 2 which is the matching level for 127.
    3. Multiply the level by the range between levels to get the closest value to the current pixel
    :param value:
    :param m:
    :return:
    '''
    # Find the range between levels
    range_between_levels = (255 / (m - 1))
    # Find the level of the current pixel
    level = round(value / range_between_levels)
    # Convert the level to an integer by flooring it
    pixel_range = math.floor(range_between_levels)
    # Return the closest value to the current pixel
    return level * pixel_range


def m_error_diffusion(file_path: str = None, m: int = -1):
    """
    Perform error diffusion on the input image
    Explanation:
    1.  Loop over the image pixels
    2.  Add the error to the current pixel
    3.  Find the closest value to the current pixel
    4.  Calculate the difference between the original and the new pixel value
    5.  Set the pixel to one of the m levels
    6.  Spread the error to the neighboring pixels according to the Floyd-Steinberg algorithm
    7.  Add the error to the last pixel in the row
    :param file_path: The name of the image file
    :param m: The number of levels
    :return: None
    """
    if file_path is None:
        raise ValueError('file_name must be provided')

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception(f'Error reading image {file_path}')
    if m < 2:
        raise ValueError('m must be greater than 1')
    # Create a copy of the original image
    original_image = image.copy()

    # Get the image dimensions
    num_of_rows, num_of_cols = image.shape
    # Initialize the error matrix
    error_matrix = np.zeros(image.shape, dtype=np.float64)
    # Convert the image to float64 values
    image = image.astype(np.float64)

    # Loop over the image pixels
    for col in range(num_of_cols - 1):
        for row in range(num_of_rows - 1):
            # Add the error to the current pixel
            image[row, col] += error_matrix[row, col]
            # Find the closest value to the current pixel
            temp = find_closest_value(image[row, col], m)
            # Calculate the difference between the original and the new pixel value
            diff = image[row, col] - temp
            # Set the pixel to either 0 or 255
            image[row, col] = temp
            # Spread the error to the neighboring pixels according to the Floyd-Steinberg algorithm
            error_matrix[row:row + 2, col:col + 2] += diff * np.array([[0, 3/8],
                                                                       [3/8, 1/8]])

    # Add the error to the last pixel in the row
    for row in range(num_of_rows - 1):
        image[row, num_of_cols - 1] += error_matrix[row, num_of_cols - 1]
        temp = 0 if image[row, num_of_cols - 1] < 128 else 255
        diff = image[row, num_of_cols - 1] - temp
        image[row, num_of_cols - 1] = temp
        error_matrix[row + 1, num_of_cols - 1] += diff * 3 / 8

    # Add the error to the last pixel
    image[num_of_rows - 1, num_of_cols - 1] = image[num_of_rows - 1, num_of_cols - 1] + error_matrix[num_of_rows - 1, num_of_cols - 1]
    # Set the last pixel to either 0 or 255
    image[num_of_rows - 1, num_of_cols - 1] = 255 if image[num_of_rows - 1, num_of_cols - 1] < 128 else 0

    # Convert the image to uint8 (8-bit unsigned integer)
    image = np.uint8(image)

    # plot the original image
    plt.imshow(original_image, cmap='gray')
    plt.show()

    # plot the image after error diffusion
    plt.imshow(image, cmap='gray')
    plt.show()

    # save the result to image directory
    current_directory = Path(file_path).parent
    file_name = Path(file_path).stem
    cv2.imwrite(f'{current_directory}\\{file_name}_{m}.png', image)
    print('The .png image was saved to the current directory')


def main():
    input_file_path = input('Enter the gray image file path: ')
    input_m = int(input('Enter the number of levels: '))
    print(f'Calculating Error Diffusion with m = {input_m}...')

    # Apply the error diffusion algorithm
    m_error_diffusion(file_path=input_file_path, m=input_m)

    print('Done')


if __name__ == '__main__':
    main()
