import numpy as np
import os
import cv2
from shutil import copy2
from tqdm import tqdm
import uuid


if __name__ == '__main__':
    dir_path = "/home/vietnhh/FTech.ai/data/data_ocr/vie/ocr_dataset"
    sub_folder = ["en_00", "en_01", "meta", "vi_00", "vi_01", "random"]

    max_length = 0

    image_ext = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    test_txt = open(os.path.join(dir_path, "full/test.txt"), 'w')
    train_txt = open(os.path.join(dir_path, "full/train.txt"), 'w')

    for folder in sub_folder:
        list_path = os.listdir(os.path.join(dir_path, folder))
        list_path = [path for path in list_path if path.lower().endswith(image_ext)]

        for i, path in tqdm(enumerate(list_path)):
            file_name, file_ext = os.path.splitext(path)

            with open(os.path.join(dir_path, folder, file_name + ".txt"), "r") as txt_file:
                label = txt_file.read()

            new_name = file_name + "_" + str(uuid.uuid4()) + file_ext

            if i % 10 == 0:
                train_test = "full/dcu_test"
                test_txt.write(f"{new_name}\t{label}\n")
            else:
                train_test = "full/dcu_train"
                train_txt.write(f"{new_name}\t{label}\n")

            copy2(os.path.join(dir_path, folder, path), os.path.join(dir_path, train_test, new_name))

            length = len(label)
            if length > max_length:
                max_length = length

    test_txt.close()
    train_txt.close()

    print("Max length = ", max_length)