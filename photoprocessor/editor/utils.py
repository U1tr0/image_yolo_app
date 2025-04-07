import cv2
import numpy as np
from django.conf import settings
import os

def apply_filter(image_path, filter_type='grayscale'):
    img = cv2.imread(image_path)
    if filter_type == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        img = cv2.GaussianBlur(img, (15, 15), 0)
    # Другие фильтры...
    output_path = os.path.join(settings.MEDIA_ROOT, 'processed', os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path