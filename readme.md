# Workpiece Chattering Detection

A GUI application for detecting and measuring chattering areas in machined workpieces using image processing techniques.

## Overview

This tool helps in identifying and measuring chattering regions in machined parts by using computer vision algorithms to detect irregularities on the surface. The application calculates the actual area of chattering in square millimeters using a reference circle of known dimensions.

## Requirements

- Tested on Python 3.8.10
- Dependencies listed in requirements.txt

## Installation

```bash
# Clone the repository (if using version control)
git clone [repository-url]
cd workpiece-chattering-detection

# Install required packages
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

### Step-by-step guide:

1. Click "Import Image" to load an image of your workpiece
2. The application will automatically try to detect the reference circle (highlighted in green)
3. Draw a rectangle around the chattering area you want to analyze, it will frame out the chattering area automatically.
4. Adjust the parameters as needed:
   - **Adaptive Threshold**: Controls the block size for binary thresholding
   - **Gabor Filter**: Adjusts texture detection sensitivity and direction
   - **Blur**: Controls the amount of smoothing applied

### Parameters Explanation
#### Adaptive Threshold
- **Block Size**: Size of the pixel neighborhood used for thresholding. Higher values consider larger areas, which can help with uneven lighting but may lose detail.

#### Gabor Filter
- **Kernel Size**: Controls the size of the Gabor filter. Larger kernels capture more texture information but require more processing.
- **Theta**: Orientation of the Gabor filter in degrees. Adjust to match the direction of chattering patterns.
- **Lambda**: Wavelength of the sinusoidal factor. Controls the spacing of detected texture patterns.

#### Blur
- **Kernel Size**: Controls the amount of blurring. Helps reduce noise but may also blur important details if set too high.

## Methodology

1. **Image Processing Pipeline**:
   - Convert image to grayscale
   - Apply adaptive thresholding to create a binary image
   - Use Gabor filters to highlight texture patterns
   - Apply Gaussian blur to reduce noise
   - Create binary mask with post-processing threshold
   - Apply morphological transformations to enhance contours
   - Find and measure the largest contour in the ROI

2. **Area Calculation**:
   - Detect reference circle in the image using Hough transform
   - Calculate the area of the detected chattering region in pixels
   - Convert to real-world measurements using the known area of the reference circle (6mm diameter)

## Project Structure
```
workpiece-chattering-detection/
├── app/
│   ├── components/
│   │   ├── ImageProcessor.py
│   │   ├── RectangleSelector.py
│   │   ├── ThetaHistogramWindow.py
│   │   └── _utils.py
│   └── app.py
├── utils/
│   ├── binarize.py
│   └── find_focus.py
├── readme.md
└── requirements.txt
```