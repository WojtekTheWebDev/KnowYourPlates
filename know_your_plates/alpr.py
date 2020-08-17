import numpy as np
import cv2
import imutils
import pytesseract
import utils
from PIL import Image
from typing import Callable


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Wrapper for OpenCV's Gaussian blur. Image is blurred with (3, 3) kernel.

    Parameters
    ----------
    image: numpy.ndarray
        Image as numpy array. Should be converted into grayscale.

    Returns
    -------
    numpy.ndarray
        Blurred image using Gaussian blur

    Contribute
    ----------
    Source: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
    """
    return cv2.GaussianBlur(image, (3, 3), 0)


def median_blur(image: np.ndarray) -> np.ndarray:
    """
    Wrapper for OpenCV's median blur. Aperture linear size for medianBlur is 3.

    Parameters
    ----------
    image: numpy.ndarray
        Image as numpy array. Should be converted into grayscale.

    Returns
    -------
    numpy.ndarray
        Blurred image using median blur

    Contribute
    ----------
    Source: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=medianblur#medianblur
    """
    return cv2.medianBlur(image, 3)


def bilateral_filter(image: np.ndarray) -> np.ndarray:
    """
    Wrapper for OpenCV's bilateral filter. Diameter of each pixel neighborhood is 11. 
    Both filter sigma in the color space and filter sigma in the coordinate space are 17.

    Parameters
    ----------
    image: numpy.ndarray
        Image as numpy array. Should be converted into grayscale.

    Returns
    -------
    numpy.ndarray
        Blurred image using bilateral filter

    Contribute
    ----------
    Source: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
    """
    return cv2.bilateralFilter(image, 11, 17, 17)


def canny(image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Wrapper for OpenCV's Canny algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Image as numpy array
    threshold1 : int
        Lower value of the threshold
    threshold2 : int
        Upper value of the threshold

    Returns
    -------
    numpy.ndarray
        Binarized image using Canny's algorithm.

    Contribute
    ----------
    Source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    """
    edged = cv2.Canny(image, threshold1, threshold2)
    return edged


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Function automatically sets up lower and upper value of the threshold
    based on sigma and median of the image

    Parameters
    ----------
    image : numpy.ndarray
        Image as numpy array
    sigma : float


    Returns
    -------
    numpy.ndarray
        Binarized image with Canny's algorithm

    Contribute
    ----------
    Source: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """
    Wrapper for OpenCV's Otsu's threshold algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Image as numpy array

    Returns
    -------
    numpy.ndarray
        Binarized image using Otsu's algorithm.

    Contribute
    ----------
    Source: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Wrapper for OpenCV's adaptive threshold algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Image as numpy array

    Returns
    -------
    numpy.ndarray
        Binarized image using adaptive threshold.

    Contribute
    ----------
    Source: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def preprocess(image: np.ndarray, new_size: tuple, blurring_method: Callable, binarization_method: Callable) -> np.ndarray:
    """
    Resizing, converting into grayscale, blurring and binarizing

    Parameters
    ----------
    image : numpy.ndarray
        Image as numpy array
    new_size  : tuple of integers
        First argument of the tuple is new width, second is the new height of the image
    blurring_method : function
        Function as an object. Suggested functions from this module: gaussian_blur, median_blur, bilateral_filter
    binarization_method : function
        Function as an object. Suggested functions from this module: threshold_otsu, adaptive_threshold, canny, auto_canny

    Returns
    -------
    numpy.ndarray
        Preprocessed image.

    Contribute
    ----------
    Grayscale conversion: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    Bilateral filter: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
    """
    if new_size is not None:
        image = imutils.resize(image, width=new_size[0], height=new_size[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = blurring_method(gray)
    binary_img = binarization_method(blurred)
    return binary_img


def plate_contours(image: np.ndarray) -> list:
    """
    Finding contours on the binarized image.
    Returns only 10 (or less) the biggest rectangle contours found on the image.

    Parameters
    ----------
    image : numpy.ndarray
       Binarized image as numpy array

    Returns
    -------
    list of numpy.ndarray
       List of found OpenCV's contours.

    Contribute
    ----------
    Finding contours: https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
    """
    cnts = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    plate_cnts = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            plate_cnts.append(approx)

    return plate_cnts


def crop_image(original_img: np.ndarray, plate_cnt: np.ndarray) -> np.ndarray:
    """
    Crops part of the image based on the OpenCV contour

    Parameters
    ----------
    original_img : numpy.ndarray
       Original image (RGB colorscale) as numpy array
    plate_cnt :  np.ndarray
        OpenCV contour

    Returns
    -------
   numpy.ndarray
       Cropped part of the original image

    Contribute
    ----------
    Bounding rectangle: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7
    """
    x, y, w, h = cv2.boundingRect(plate_cnt)
    return original_img[y:y + h, x:x + w]


def prepare_ocr(image: np.ndarray) -> np.ndarray:
    """
    Prepares image to the OCR process by resizing and filtering (for noise reduction)

    Parameters
    ----------
    image : numpy.ndarray
       Image as numpy array

    Returns
    -------
    numpy.ndarray
       Image prepaired to the OCR process

    Contribute
    ----------
    Resizing: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20resize(InputArray%20src,%20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,%20int%20interpolation)
    Bilateral filter: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
    """
    image = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    image = cv2.bilateralFilter(image, 11, 17, 17)
    return image


def ocr(img_path: str) -> str:
    """
    Wrapper for Tesseract image_to_string function

    Parameters
    ----------
    img_path : str
       Path to the image

    Returns
    -------
    str
       Text recognized on the image

    Contribute
    ----------
    PyTesseract: https://pypi.org/project/pytesseract/
    """
    return pytesseract.image_to_string(Image.open(img_path))


def license_plate_recognition(img_path: str, new_size: tuple, blurring_method: Callable, binarization_method: Callable) -> str:
    """
    Automatic license plate recognition algorithm.
    Found license plate is stored in ./results/ directiory as license_plate.jpg

    Parameters
    ----------
    img_path : str
       Path to the image
    new_size  : tuple of integers
        First argument of the tuple is new width, second is the new height of the image
    blurring_method : function
        Function as an object. Suggested functions from this module: gaussian_blur, median_blur, bilateral_filter
    binarization_method : function
        Function as an object. Suggested functions from this module: threshold_otsu, adaptive_threshold, canny, auto_canny

    Returns
    -------
    str
       Text recognized on the image
    """
    image = cv2.imread(img_path)
    binary_img = preprocess(
        image,
        new_size,
        blurring_method,
        binarization_method
    )
    plate_cnts = plate_contours(binary_img)
    if len(plate_cnts) == 0:
        return ''
    for c in plate_cnts:
        cropped = crop_image(image, c)
        cropped = prepare_ocr(cropped)
        img_path = utils.save_image_plt(
            "./results",
            'license_plate',
            cropped
        )
        recognized_txt = utils.remove_special_chars(ocr(img_path))
        if len(recognized_txt) > 0:
            return recognized_txt
    return ''
