import os
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def threshold(image: np.ndarray, threshold_type: int=cv.THRESH_OTSU) -> np.ndarray:
    """
    Talking about Otsu's thresholding method here.

    :param image: np.ndarray -> Input image of playing card
    :param threshold_type: int -> Type of thresholding method to use, default is Otsu's
    :return thresholded_image: np.ndarray -> Binary Thresholded image
    """

    _, thresholded_image = cv.threshold(image, 0, 255, threshold_type)

    return thresholded_image

def find_largest_contour(thresholded_image: np.ndarray) -> tuple:
    """
    Finds the largest contour in the thresholded image

    :param thresholded_image: input thresholded image
    :return largest_contour, contours: tuple -> the largest contour and all contours in the image
    """
    contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    return largest_contour, contours



def draw_contour(image: np.ndarray, contour: np.ndarray):
    """
    Draws the contour on the normal image
    :param image: np.ndarray -> input image
    :param contour: np.ndarray -> contour to draw
    :return image_copy: np.ndarray -> image with contour drawn
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image_copy = image.copy()
    cv.drawContours(image_copy, [contour], -1, (0, 255, 0), 5)
    return image_copy



def calculate_orientation(largest_contour: np.ndarray):
    """
    Calculates the orientation(angle) of the largest contour via PCA analysis
    :param largest_contour: np.ndarray -> largest contour in the image
    :return: angle: float -> angle of the largest contour
    """
    data = largest_contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv.PCACompute(data, mean=None)[:2]
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    return angle



def rotate_to_vertical(image: np.ndarray, orientation: float):
    """
    Rotates the original image to the vertical
    :param image: np.ndarray -> input image
    :param orientation: float -> orientation of the image
    :return rotated_image: np.ndarray -> rotated image
    """

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
    """
    Shows the original and result image(after rotation) with histograms
    :param gray_image:
    :param result:
    :param kwargs:
    :return:
    """
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

def get_mask(thresholded_rotated_image, contour):
    mask = np.zeros(thresholded_rotated_image.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, -1)
    return mask

def crop_image(image, mask):
    return cv.bitwise_and(image, image, mask=mask)


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
        thresh_rotated = threshold(rotated_image, threshold_type=cv.THRESH_BINARY)
        mask = get_mask(thresh_rotated, largest_contour_rotated)
        rotated_image = crop_image(rotated_image, mask)

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