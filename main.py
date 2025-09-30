import os
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sobel_kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_kernel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

prewitt_kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitt_kernel_y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

laplacian_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])


def threshold(gray_image: np.ndarray, threshold_type: int=cv.THRESH_OTSU):
    _, thresholded_image = cv.threshold(gray_image, 0, 255, threshold_type)

    return thresholded_image

def find_largest_contour(thresholded_image: np.ndarray):
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    return largest_contour, contours

def draw_contour(image: np.ndarray, contour: np.ndarray):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image_copy = image.copy()
    cv.drawContours(image_copy, [contour], -1, (0, 255, 0), 5)
    return image_copy

def calculate_orientation(largest_contour: np.ndarray):
    data = largest_contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv.PCACompute(data, mean=None)[:2]
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    return angle



def rotate_to_vertical(image: np.ndarray, orientation: float):
    # Normalize angle to [-90, 90]; this ensures horizontal shapes are recognized

    orientation = -orientation % 180
    # Compute rotation to nearest vertical (90 degrees)
    angle_to_rotate = 90 - orientation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle_to_rotate, 1.0)
    result = cv.warpAffine(image, M, (w, h))
    return result.astype(np.uint8)

def show_image(gray_image: np.ndarray, result: np.ndarray=None, **kwargs):
    if result is None:
        print("result is None in show_image, skipping display.")
        return

    orientation = kwargs.get("orientation", None)
    final_orientation = kwargs.get("final_orientation", None)


    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax_img_orig, ax_hist_orig = axes[0]
    ax_img_thr, ax_hist_thr = axes[1]

    # Histogram for original image
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

    # Show original image
    ax_img_orig.imshow(gray_image, cmap="gray")
    ax_img_orig.set_title(f"(Original), Orientation: {orientation}")
    ax_img_orig.axis("off")

    # Show histogram of original
    ax_hist_orig.plot(hist, color="black")
    ax_hist_orig.set_title("Histogram (Original)")
    ax_hist_orig.set_xlim([0, 256])
    ax_hist_orig.set_ylabel("Number of pixels")
    ax_hist_orig.set_xlabel("Pixel intensity")
    ax_hist_orig.grid(True)


    # Histogram for result image
    hist_result = cv.calcHist([result], [0], None, [256], [0, 256])

    ax_img_thr.imshow(result, cmap="gray")
    ax_img_thr.set_title(f"(Final), Orientation: {final_orientation}")
    ax_img_thr.axis("off")

    ax_hist_thr.plot(hist_result, color="black")
    ax_hist_thr.set_title("Histogram (Final)")
    ax_hist_thr.set_xlim([0, 256])
    ax_hist_thr.set_ylabel("Number of pixels")
    ax_hist_thr.set_xlabel("Pixel intensity")
    ax_hist_thr.grid(True)

    ax_img_thr.axis("off")
    ax_hist_thr.axis("off")

    plt.tight_layout()
    plt.show()



def main(args):

    image_path = args.image_path
    images = [file for file in os.listdir(image_path)]

    counter = 1

    for image in images:

        print(f"{'='*15} Processing Testimage{counter} {'='*15}\n")

        image = cv.imread(os.path.join(image_path, image), cv.IMREAD_GRAYSCALE)
        thresholded_image = threshold(image)
        largest_contour, contours = find_largest_contour(thresholded_image)
        orientation = calculate_orientation(largest_contour)
        rotated_image = rotate_to_vertical(image, orientation)
        largest_contour_rotated, contours = find_largest_contour(rotated_image)
        final_orientation = calculate_orientation(largest_contour_rotated)
        show_image(image, rotated_image, orientation=orientation, final_orientation=final_orientation)


        print(f" -> Initial Orientation: {orientation} degrees")
        print(f" -> Final Orientation: {final_orientation} degrees")

        print(f"\n{'='*15} Finished Processing Testimage{counter-1} {'='*15}")
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with edge detection")
    parser.add_argument("image_path", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    main(args)