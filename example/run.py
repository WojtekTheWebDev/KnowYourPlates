# run.py
import argparse
from knowyourplates import alpr

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

recognized_text = alpr.license_plate_recognition(
    img_path=args['image'],
    new_size=None,
    blurring_method=alpr.bilateral_filter,
    binarization_method=alpr.adaptive_threshold
)
print(recognized_text)
