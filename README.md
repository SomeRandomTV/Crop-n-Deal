# Crop n Deal

---

**ECE 4367** Image Processing Project 2  
*Author: Turd "2 ball" Ferguson*

A Python script to rotate images of playing cards to be vertical and crop the image so it is just the card.

**TODO**: Fine tune the angle then crop

---

## Overview

This script takes a directory containing only images as a command line argument and processes each image  
so the card is at anyangle showing just the card.

**NOTE:** Both the before and after processing image will be displayed side-by-side, with orientation shown for each.

---

## The Problem

Write a script that automatically localizes and undoes the rotation of a playing card, leaving it
upright and cropped.

**Restrictions and specifications:**
1. Only use predefined functions for image I/O (`cv.imread`), display (`matplotlib`), and geometric transformations (`cv.warpAffine`).
2. No user input except for the input filename.
3. Display both input and output pairs as shown in the example.
4. Must successfully process three provided images. Additional images recommended for testing.

---

## Algorithm Overview

### 1. Thresholding 
Convert the image to grayscale, then binarize using Otsu's thresholding:
```python
    _, thresholded_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
```
This separates card content from background.

### 2. Contour Detection
Find contours in the binary image, and return the largest contour (by area):

```python
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
```


### 3. Calculate Orientation 
Use PCA on the largest contour to find its orientation:

```python
    data = largest_contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv.PCACompute(data, mean=None)[:2]
    angle = np.arctan2(eigenvectors, eigenvectors) * 180 / np.pi    
```

This angle is relative to the horizontal. Vertical is ±90 degrees. We normalize/rotate so the card is upright.

### 4. Rotate Image
Rotate the image such that the card appears vertical. The code for normalization and rotation:

```python
    orientation = -orientation % 180
    angle_to_rotate = 90 - orientation
    M = cv.getRotationMatrix2D(center, angle_to_rotate, 1.0)
    result = cv.warpAffine(image, M, (w, h))
```


### 5. Visualization
Both original and result images are shown, with histograms and the card's orientation:

```python
    show_image(image, rotated_image, orientation=orientation, final_orientation=final_orientation)
```

--

## Usage

```bash
  python main.py <directory_path>
```

**Example:**
```bash
    python main.py ./Testimages/
```


**Requirements:**
- Images should contain a single playing card.
- Cards can be at any angle or perspective.

---

## Dependencies

```python
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt
```
---

## Processing Pipeline

1. Read image as grayscale
2. Threshold with Otsu's method
3. Find largest contour
4. Get orientation using PCA
5. Rotate to vertical
6. Show before & after images, with histograms and orientation values

---

## Testing

**Expected Output:**
- Original and processed images displayed side-by-side (with orientation indicated)
- Console output showing image filenames and orientation angles

---

## File Structure

crop_n_deal/
├── main.py
├── README.md
└── Testimages/
    ├── Testimage1.jpg
    ├── Testimage2.jpg
    └── Testimage3.jpg


