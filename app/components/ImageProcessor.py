import cv2
import numpy as np

class ImageProcessor:

    imshow_disable = False

    def __init__(self, image_path):
        self.image_path = image_path

    @staticmethod
    def process(image, **kwargs):
        if image is None:
            print("No image loaded")
            return None
        
        # Get parameters
        block_size = kwargs.get('block_size', 7) # binarize

        gabor_ksize = kwargs.get('gabor_ksize', 5) # gabor
        theta = kwargs.get('theta', 7) # gabor
        lambd = kwargs.get('lambd', 7) # gabor

        blur_ksize = kwargs.get('blur_ksize', 21) # blur

        post_binarize_th = kwargs.get('post_binarize_th', 130) # post binarize

        morph_times = kwargs.get('morph_times', 3) # morph
        morph_kernel_size = kwargs.get('morph_kernel_size', 5) # morph
        
        # Convert to grayscale
        p_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        p_img = cv2.adaptiveThreshold(
            p_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)

        p_img = ImageProcessor.gabor_filter(p_img, ksize=gabor_ksize, sigma=0.1, theta=theta, lambd=lambd)
        p_img = ImageProcessor.blur(p_img, kernel_size=blur_ksize)
        p_img = ImageProcessor.post_binarize(p_img, threshold=post_binarize_th)
        p_img = ImageProcessor.morphological_transform(p_img, times=morph_times, kernel_size=morph_kernel_size)
        max_contours = ImageProcessor.find_max_region(p_img)

        return max_contours
    
    @staticmethod
    def gabor_filter(img, ksize=3, sigma=1.0, theta=0, gamma=0.4, lambd=10.0):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kernel /= 2 * np.pi * sigma**2
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        return filtered_img
    
    @staticmethod
    def blur(img, kernel_size=3):
        # kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        kernel = np.outer(kernel, kernel)  # Create a 2D Gaussian kernel
        kernel /= np.sum(kernel)  # Normalize the kernel

        filtered_img = cv2.filter2D(img, -1, kernel)
        return filtered_img
    
    @staticmethod
    def post_binarize(img, threshold=127):
        _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary_img

    @staticmethod
    def morphological_transform(img, times=3, kernel_size=5):
        # shrink and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for i in range(times):
            img = cv2.erode(img, kernel, iterations=3)
            img = cv2.dilate(img, kernel, iterations=3)
        return img

    @staticmethod
    def find_max_region(bin_img):
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        
        return max_contour

    @staticmethod
    def scalable_imshow(img, scale=0.5, msg="Processed Image"):

        if ImageProcessor.imshow_disable:
            return
        
        imgS = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imshow(msg, imgS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def detect_circles(
        img: np.ndarray,
        minRadius: int = 50,
        maxRadius: int = 300,
        param2: int = 100,
        resize_factor: float = 0.5,
    ):
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
        img = clahe.apply(img)
        img = cv2.GaussianBlur(img, (5, 5), 1.5, 1.5)
        img = cv2.Canny(img, 100, 200, apertureSize=3)
        img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=2)
        img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=2)

        circles = cv2.HoughCircles(
            image=img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2*minRadius, param1=100, 
            param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )

        if circles is None:
            return None
        
        return circles[0] / resize_factor



if __name__ == "__main__":
    # test_img = cv2.imread("G:/Cellery/fan_vision_reg/data/S__281157640_0.jpg")
    test_img = cv2.imread("G:/Cellery/fan_vision_reg/data/S__281354254_0.jpg")
    
    x, y, w, h = 100, 759, 400, 600  # Example coordinates for cropping
    cropped_image = test_img[y:y+h, x:x+w]
    p_img = ImageProcessor.process(cropped_image)