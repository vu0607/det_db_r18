import os
import cv2
import ast
import argparse
from tqdm import tqdm

from paddleocr import PaddleOCR
from ppocr.metrics.det_metric import DetMetric
from ppocr.metrics.det_metric_evaluation import DetMetricEvaluation


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        default="/home/nhhviet/ftech.ai/data/ekyc/private_test",
        type=str,
        help="Path to Evaluation data dir",
    )
    parser.add_argument(
        "-l",
        "--label_file_list",
        default="/home/nhhviet/ftech.ai/data/ekyc/private_test/annotations.txt",
        type=str,
        help="Path to annotaions files",
    )
    return parser.parse_args()


def format_label(label):
    label = ast.literal_eval(label)

    label_list = []
    ignore_list = []
    transcription_list = []
    kie_class_list = []
    for ins in label:
        label_list.append(ins["points"])
        ignore_tag = True if ins["transcription"] == '###' else False
        ignore_list.append(ignore_tag)
        kie_class_list.append([ins["kie_class"]])
        transcription_list.append([ins["transcription"]])

    return [[], [], [label_list], [ignore_list], transcription_list, kie_class_list]


def main():
    args = arg_parser()
    img_dir = args.data_dir
    annotation_path = args.label_file_list

    # build model
    ocr = PaddleOCR(
        use_gpu=True,
        module="ekyc",
        det=True,
        rec=False,
        logging_time=True,
    )
    # build metric
    metric = DetMetricEvaluation(iou_values=[0.5, 0.6, 0.7, 0.8], save_res_path="test_result/det_db/predicts_db.txt")

    with open(annotation_path, 'r') as f:
        data = f.readlines()

    for line in tqdm(data):
        path, label = line.strip("\n").split("\t")
        label = format_label(label)
        img = cv2.imread(os.path.join(img_dir, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = ocr.ocr(img, rec=False, det=True)
        metric([{'points': pred}], label, [path], img_dir)
    print(metric.get_metric())
    ocr.text_detector.autolog.report()


if __name__ == '__main__':
    main()
