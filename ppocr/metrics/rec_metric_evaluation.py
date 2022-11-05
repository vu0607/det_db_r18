# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path
import shutil

import Levenshtein
import cv2

from pathlib import Path

from ppocr.utils.image_processing import SaveTestAnnotation
from ppocr.postprocess.ekyc_postprocess import EKYCPostProcess


class RecMetricEvaluation(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()
        self.true_dir = "test_result/true"
        self.false_dir = "test_result/false"
        self.acceptable_dir = "test_result/acceptable"

        if 'kie_class' in kwargs.keys():
            self.kie_class = kwargs['kie_class']
            self.kie_metric = {}
            for cls in self.kie_class:
                self.kie_metric[cls + '_true'] = 0
                self.kie_metric[cls + '_cer'] = 0
                self.kie_metric[cls + '_total'] = 0
        else:
            self.kie_class = None

        if os.path.isdir(self.true_dir):
            shutil.rmtree(self.true_dir)
        Path(self.true_dir).mkdir(parents=True, exist_ok=True)

        if os.path.isdir(self.false_dir):
            shutil.rmtree(self.false_dir)
        Path(self.false_dir).mkdir(parents=True, exist_ok=True)

        if os.path.isdir(self.acceptable_dir):
            shutil.rmtree(self.acceptable_dir)
        Path(self.acceptable_dir).mkdir(parents=True, exist_ok=True)

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0

        batch_file_name = args[1]
        dir_path = args[2]

        for (pred, pred_conf), (target, _), file_name in zip(preds, labels, batch_file_name):
            org_pred, org_target = pred, target

            pred = EKYCPostProcess()(pred)

            if self.kie_class is not None:
                cls = self.get_cls_name(file_name)
            else:
                cls = None

            pred = strictly_pred = pred.replace(" ", "").lower()
            target = strictly_target = target.replace(" ", "").lower()

            if self.kie_class is not None:
                pred, target = self.ekyc_acceptable(pred, target, cls)

            edit_dis = Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            norm_edit_dis += edit_dis

            if self.kie_class is not None:
                if cls is None:
                    print(f"WARNING: {file_name} not in kie class")
                else:
                    self.get_kie_metric(cls, edit_dis)

            ann = SaveTestAnnotation(
                "resources/fonts/times.ttf", 32)

            if pred == target:
                correct_num += 1
                if strictly_pred == strictly_target:
                    img = cv2.imread(os.path.join(dir_path, file_name))
                    cv2.imwrite(os.path.join(self.true_dir, file_name), img)
                else:
                    ann(
                        dir_path,
                        file_name,
                        self.acceptable_dir,
                        org_pred,
                        org_target,
                    )
            else:
                    ann(
                        dir_path,
                        file_name,
                        self.false_dir,
                        org_pred,
                        org_target,
                    )
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': norm_edit_dis / all_num
        }

    def get_kie_metric(self, cls, edit_dis):
        if edit_dis == 0:
            self.kie_metric[cls + '_true'] += 1
        else:
            self.kie_metric[cls + '_cer'] += edit_dis
        self.kie_metric[cls + '_total'] += 1

    def get_cls_name(self, file_name):
        for cls in self.kie_class:
            if cls in file_name:
                return cls
        return

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / self.all_num
        norm_edit_dis = self.norm_edit_dis / self.all_num
        self.reset()

        metric = {}
        if self.kie_class is not None:
            metric.update(self.calculate_kie_class_metric())

        metric['acc'] = acc
        metric['norm_edit_dis'] = norm_edit_dis

        return metric

    def calculate_kie_class_metric(self):
        metric = {}
        total_wo_other = 0
        true_wo_other = 0
        cer_wo_other = 0

        for cls in self.kie_class:
            cls_total = self.kie_metric[cls + "_total"]
            if cls_total != 0:
                metric[cls + "_acc"] = self.kie_metric[cls + "_true"] / cls_total
                metric[cls + "_cer"] = self.kie_metric[cls + "_cer"] / cls_total

                if cls != "Other":
                    total_wo_other += cls_total
                    true_wo_other += self.kie_metric[cls + "_true"]
                    cer_wo_other += self.kie_metric[cls + "_cer"]

            else:
                metric[cls + "_acc"] = None
                metric[cls + "_cer"] = None

        metric['acc_wo_other'] = true_wo_other / total_wo_other
        metric['acc_wo_cer'] = cer_wo_other / total_wo_other

        return metric

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0

    @staticmethod
    def ekyc_acceptable(pred, target, cls):
        if pred[-1] in ['"', ":", ".", "-", ","]:
            pred = pred[:-1]
        if target[-1] in ['"', ":", ".", "-", ","]:
            target = target[:-1]

        pred = pred.replace("tp.", "tp")
        target = target.replace("tp.", "tp")

        if cls == "PlaceOfIssue_Value":
            pred = pred.replace("giámđốcca", "")
            target = target.replace("giámđốcca", "")

        if cls == "PlaceOfIssue_Value" or cls == "PlaceOfOrigin_Value" or cls == "PlaceOfResidence_Value":
            pred = pred.replace("-", "")
            target = target.replace("-", "")

        if cls == "Feature_Value":
            # For cccd_v1_back
            pred = pred.replace("đặcđiểmnhậndạng", "")

            # Convert characters
            pred = pred.replace(";", ".")
            pred = pred.replace(":", ".")
            pred = pred.replace(",", ".")
            target = target.replace(";", ".")
            target = target.replace(":", ".")
            target = target.replace(",", ".")

            index = [pos for pos, char in enumerate(pred) if char == "."]
            for i in index:
                if i != -1 and pred[i-1] == "c" and pred[i+1].isnumeric():
                    pred = pred[:i] + pred[i+1:]

            index = [pos for pos, char in enumerate(target) if char == "."]
            for i in index:
                if i != -1 and target[i - 1] == "c" and target[i + 1].isnumeric():
                    target = target[:i] + target[i + 1:]

        if cls == "PlaceOfResidence_Value":
            pred = pred.replace("nơithườngtrú:", "")
            pred = pred.replace("nơithườngtrú", "")
            pred = pred.replace("nơithườngtr", "")

        if cls == "PlaceOfOrigin_Value":
            pred = pred.replace("nguyênquán:", "")
            pred = pred.replace("nguyênquán", "")

            pred = pred.replace("quêquán:", "")

        if cls == "IDNumber_Value":
            pred = pred.replace("số:", "")
            pred = pred.replace("số", "")

        return pred, target