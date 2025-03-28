import cv2
import numpy as np
from numba import jit

def select_in_focus_region(gray, threshold=50):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_map = np.abs(laplacian)
    sharpness_map = cv2.GaussianBlur(sharpness_map, (5,5), 0)
    sharpness_map = np.uint8(255 * sharpness_map / np.max(sharpness_map))
    binary = np.zeros_like(sharpness_map, dtype=np.uint8)
    binary[sharpness_map > threshold] = 255

    # dialate and erode

    # diamond kernel
    kernel = diamond_kernel(3)
    binary = auto_morph(binary, kernel, 30)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binary

def diamond_kernel(size):
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if i + j < size // 2 or i + j >= size + size // 2:
                kernel[i, j] = 0
            else:
                kernel[i, j] = 255
    return kernel

def auto_morph(binary, kernel, iterations):
    iter = 1
    for _ in range(iterations):
        for t in ['erode', 'dilate']:
            if t == 'erode':
                binary = cv2.erode(binary, kernel, iterations=iter)
            else:
                binary = cv2.dilate(binary, kernel, iterations=iter)
            iter += 1

        print(loss(binary))
    return binary

def loss(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Get areas of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the background (label 0)
    n = len(areas)  # Number of islands
    
    if n == 0:
        return 0  # No islands, loss is 0
    
    # Proportion of the largest area
    largest_area = max(areas)
    total_area = binary.size
    p = largest_area / total_area
    
    # Compute loss
    alpha = 0.1  # Adjust alpha as needed
    return alpha * n - p + 1