# Crop n Deal

---

**ECE 4367** Image Processing Project 2 \
*Author: Turd "2 ball" Ferguson*

A Python script to rotate images of playing cards to be vertical and crop the image so it is just the card.

---

## Overview

This script takes a directory containing only .jpg images as a command line argument and processes each image \
so the card is vertical and showing just the card

**NOTE:** Both the before and after processing image must be displayed

---

## The Problem

Write a script that can automatically localize and undo the rotation of a playing card, leaving it
upright and cropped.

Script must follow these restrictions and specifications:
1. It may only use predefined functions for image I/O (e.g., Matlab/OpenCV imread), display (e.g.,
Matlab/OpenCV imshow) and geometric transformations (e.g., Matlab/OpenCV imrotate). Use of other
functions may be allowed after a case-by-case consideration.
2. Except for the input filename, it may not ask for any other user input.
3. It must display the input and output pairs as shown in the example below.
4. It must successfully process the three test images that are provided. Processing other test
images is highly recommended.

---

## Algorithm Overview

The script implements a multi-step approach to detect, rotate, and crop playing cards:

### 1. **Edge Detection and Preprocessing**
   - Convert image to grayscale
   - Apply Gaussian blur to reduce noise
   - Use Canny edge detection to find card boundaries
   - Apply morphological operations to clean up edges

### 2. **Contour Detection**
   - Find contours in the edge-detected image
   - Filter contours by area to identify potential card candidates
   - Select the largest rectangular contour as the playing card

### 3. **Corner Detection and Perspective Correction**
   - Identify the four corners of the card contour
   - Order corners in a consistent manner (top-left, top-right, bottom-right, bottom-left)
   - Calculate perspective transformation matrix
   - Apply perspective correction to straighten the card

### 4. **Rotation Detection**
   - Analyze card orientation using corner positions
   - Determine rotation angle needed to make card vertical
   - Apply rotation transformation

### 5. **Final Cropping**
   - Remove excess background around the card
   - Ensure consistent output dimensions
   - Apply final cleanup and filtering

---

## Usage

```bash
python crop_n_deal.py <directory_path>
```

### Example:
```bash
python crop_n_deal.py ./test_images/
```

**Requirements:**
- Directory must contain only .jpg images
- Images should contain a single playing card
- Cards can be at any angle or perspective

---

## Dependencies

```python
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
```

**Installation:**
```bash
  pip install opencv-python numpy matplotlib 
```

---

## Algorithm Flowchart

```
Start
  ↓
Load Image from Directory
  ↓
Convert to Grayscale
  ↓
Apply Gaussian Blur
  ↓
Canny Edge Detection
  ↓
Morphological Operations
  ↓
Find Contours
  ↓
Filter by Area → [Contour too small?] → Yes → Try Next Contour
  ↓ No
Approximate Contour to Rectangle
  ↓
[4 corners found?] → No → Try Next Contour
  ↓ Yes
Order Corner Points
  ↓
Calculate Perspective Transform
  ↓
Apply Perspective Correction
  ↓
Detect Card Orientation
  ↓
Calculate Rotation Angle
  ↓
Apply Rotation Transform
  ↓
Crop to Card Boundaries
  ↓
Display Before/After Results
  ↓
[More images?] → Yes → Load Next Image
  ↓ No
End
```

---

## Testing

**Expected Output:**
- Original image displayed on the left
- Processed (rotated and cropped) card on the right
- Console output showing processing steps

---

## File Structure

```
crop_n_deal/
├── main.py                 # Main script
├── README.md               # This file
└── ImageSet2/            # Test image directory
    ├── card1.jpg
    ├── card2.jpg
    └── card3.jpg
```

---
