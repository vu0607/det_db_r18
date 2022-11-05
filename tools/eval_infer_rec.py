import os
import cv2
import argparse
from tqdm import tqdm

from paddleocr import PaddleOCR
from ppocr.metrics.rec_metric import RecMetric
from ppocr.metrics.rec_metric_evaluation import RecMetricEvaluation


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        default="instance_images",
        type=str,
        help="Path to Evaluation data dir",
    )
    parser.add_argument(
        "-l",
        "--label_file_list",
        default="instance_images/test.txt",
        type=str,
        help="Path to annotaions files",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=6,
        type=int,
        help="Batch size",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    img_dir = args.data_dir
    annotation_path = args.label_file_list
    batch_size = args.batch_size

    # build model
    ocr = PaddleOCR(
        use_gpu=True,
        module="ekyc",
        rec_batch_num=batch_size,
        logging_time=True,
    )
    # build metric
    metric = RecMetricEvaluation()

    with open(annotation_path, 'r') as f:
        data = f.readlines()

    img_num = len(data)
    label_batch = []
    pred_batch = []
    path_batch = []

    for beg_img_no in tqdm(range(0, img_num, batch_size)):
        end_img_no = min(img_num, beg_img_no + batch_size)

        img_batch = []
        for line in data[beg_img_no: end_img_no]:
            path, label = line.strip("\n").split("\t")
            img = cv2.imread(os.path.join(img_dir, path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_batch.append(img)
            label_batch.append([label, 1])
            path_batch.append(path)

        pred_batch += ocr.ocr(img_batch)
    m = metric([pred_batch, label_batch], None, path_batch, img_dir)
    print(m)
    ocr.text_recognizer.autolog.report()


if __name__ == '__main__':
    main()
