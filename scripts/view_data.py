import os
from tqdm import tqdm

import numpy as np
import cv2


def letterbox(img, new_shape=(640, 640), color=(255, 255, 255), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    dir_path = "/home/vietnhh/FTech.ai/data/data_ocr/private/printed_typical"
    image_ext = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    # list_path = os.listdir(dir_path)
    # list_path = [path for path in list_path if path.lower().endswith(image_ext)]
    list_path = ["printed_scanned_undefinedsubject_vi_typical_156_text_box_1.jpg"]

    for i, path in tqdm(enumerate(list_path)):
        file_name, file_ext = os.path.splitext(path)
        img = cv2.imread(os.path.join(dir_path, path))
        print(os.path.join(dir_path, path))

        cv2.imshow('img', letterbox(img, (32, 512), auto=True)[0])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
