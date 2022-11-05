import os
import json
import random
import shutil
from tqdm import tqdm
from pathlib import Path


def convert_paddledet_format(image_annotations):
    paddle_annotation = []

    for annotation in image_annotations:
        x, y, w, h = [int(b) for b in annotation['bbox']]
        points = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]

        transcription = annotation['attributes']['value']
        if transcription == "":
            transcription = "###"

        paddle_annotation.append({
            "transcription": transcription,
            "points": points
        })
    return paddle_annotation


def copy_image_to_output_dir(image, image_dir, output_dir):
    base_name = os.path.basename(image['file_name'])

    source = os.path.join(image_dir, image['file_name'])
    destination = os.path.join(output_dir, base_name)
    destination_dir = os.path.dirname(destination)

    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir)

    shutil.copyfile(source, destination)


def get_image_annotation(image, annotations):
    return [item for item in annotations if item['image_id'] == image['id']]


def main():
    root_path = Path("/home/nhhviet/ftech.ai/data/ekyc/training_data/v1.0.1")
    list_card_type = os.listdir(root_path)
    version = "paddle_det_v1.0.1"
    output_dir = root_path / version
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(output_dir / 'train')
    os.makedirs(output_dir / 'test')

    f_out_train = open(os.path.join(output_dir, 'train.txt'), "w")
    f_out_test = open(os.path.join(output_dir, 'test.txt'), "w")

    for card_type in list_card_type:
        if card_type != version:
            annotation_path = root_path / card_type / "anns/ann_part_1.json"
            image_dir = root_path / card_type / "images"

            with open(annotation_path, 'r') as f:
                data = json.load(f)

            images = data['images']
            random.shuffle(images)
            annotations = data['annotations']

            for i, image in tqdm(enumerate(images)):
                image_annotation = get_image_annotation(image, annotations)
                label = convert_paddledet_format(image_annotation)
                base_name = os.path.basename(image['file_name'])

                if i % 10 != 0:
                    copy_image_to_output_dir(image, image_dir, output_dir / 'train')
                    f_out_train.write(f"train/{base_name}\t{json.dumps(label, ensure_ascii=False)}\n")
                else:
                    copy_image_to_output_dir(image, image_dir, output_dir / 'test')
                    f_out_test.write(f"test/{base_name}\t{json.dumps(label, ensure_ascii=False)}\n")

            print("Successfully")


if __name__ == '__main__':
    main()