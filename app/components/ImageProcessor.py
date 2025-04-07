import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:

    imshow_disable = True

    def __init__(self, image_path):
        self.image_path = image_path

    @staticmethod
    def process(image, **kwargs):
        if image is None:
            print("No image loaded")
            return None
        
        # Get parameters
        block_size = kwargs.get('block_size', 11) # binarize
        c_value = kwargs.get('c_value', 2) # binarize
        threshold_sobel = kwargs.get('threshold_sobel', 100) # hough
        threshold_hough = kwargs.get('threshold_hough', 100) # hough
        minLineLength = kwargs.get('minLineLength', 50) # hough
        maxLineGap = kwargs.get('maxLineGap', 10) # hough
        
        # Convert to grayscale
        p_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        p_img = cv2.adaptiveThreshold(
            p_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value)
        

        # Use Sobel operator to enhance texture edges
        

        lines = ImageProcessor.hough_line_detect(
            p_img, 
            threshold_sobel=threshold_sobel, 
            threshold_hough=threshold_hough,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap
        )

        imgS = cv2.cvtColor(p_img, cv2.COLOR_GRAY2BGR)
        thetas = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                theta = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                thetas.append(theta)
                cv2.line(imgS, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        ImageProcessor.scalable_imshow(imgS, scale=0.5)

        # plot histogram of angles
        if not ImageProcessor.imshow_disable:
            plt.hist(thetas, bins=180, range=(-90, 90), color='blue', alpha=0.7)
            plt.title("Histogram of Angles")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Frequency")
            plt.grid()
            plt.show()    

        return imgS
    
    @staticmethod
    def sobel_edge_detect(img, ksize=3, threshold=None):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        p_img = np.sqrt(sobel_x**2 + sobel_y**2)
        p_img = np.uint8(255 * p_img / np.max(p_img))

        if threshold is not None:
            p_img = cv2.threshold(p_img, threshold, 255, cv2.THRESH_BINARY)[1]

        return p_img
    
    @staticmethod
    def hough_line_detect(img, threshold_sobel=100, threshold_hough=100, minLineLength=50, maxLineGap=10):
        p_img = ImageProcessor.sobel_edge_detect(img, ksize=3, threshold=threshold_sobel)
        ImageProcessor.scalable_imshow(p_img, scale=0.5, msg="Sobel Edge Detection")
        lines = cv2.HoughLinesP(p_img, 1, np.pi/180, threshold_hough, minLineLength, maxLineGap)
        return lines
    
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
        param1: int = 100,
        param2: int = 75,
        dp: int = 1,
    ):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 11)
        edges = cv2.Canny(gray, 75, 150, apertureSize=3)

        circles = cv2.HoughCircles(
            image=edges, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=2*minRadius, param1=param1, 
            param2=param2,minRadius=minRadius, maxRadius=maxRadius
        )

        if circles is None:
            return None
        
        return circles[0]



if __name__ == "__main__":
    # test_img = cv2.imread("G:/Cellery/fan_vision_reg/data/S__281157640_0.jpg")
    test_img = cv2.imread("G:/Cellery/fan_vision_reg/data/S__281354254_0.jpg")
    
    x, y, w, h = 100, 759, 400, 600  # Example coordinates for cropping
    cropped_image = test_img[y:y+h, x:x+w]
    p_img = ImageProcessor.process(cropped_image, blur_size=5, block_size=9, c_value=7, ksize=3)