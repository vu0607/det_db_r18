# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(__dir__, ''))

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path

from tools.infer import predict_system
from tools.program import load_config
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read_gif, get_image_file_list
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import OCRSystem, save_structure_res

__all__ = [
    'PaddleOCR',
    'PPStructure',
    'draw_ocr',
    'draw_structure_result',
    'save_structure_res',
    'download_with_progressbar'
]

SUPPORT_DET_MODEL = ['DB']
VERSION = '3.12.0'
SUPPORT_REC_MODEL = ['SRN']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

FSCHOOL_DEFAULT_MODEL_VERSION = '3.5.0'
EKYC_DEFAULT_MODEL_VERSION = '1.8.1'

DEFAULT_MODEL_VERSION = ".".join(["fschool", FSCHOOL_DEFAULT_MODEL_VERSION])

MODEL_URLS = {
    'fschool.2.8.1': {
        'det': {
            'vi': {
                'url': '',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v2.8.1-e8541c67/v2.8.1.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'fschool.3.2.1': {
        'det': {
            'vi': {
                'url': '',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.2.1-0503b6c0/v3.2.1.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'fschool.3.4.4': {
        'det': {
            'vi': {
                'url': '',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.4.4-33fe9395/v3.4.4.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'fschool.3.4.5': {
        'det': {
            'vi': {
                'url': '',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.4.5-5bbcba02/v3.4.5.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.4.5-5bbcba02/v3.4.5_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'fschool.3.5.0': {
        'det': {
            'vi': {
                'url': '',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.5.0-4751af85/v3.5.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-v3.5.0-4751af85/v3.5.0_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.0.0': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.0.0-01fd67f6/det_ekyc_v1.0.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.0.0-01fd67f6/rec_ekyc_v1.0.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.1.0': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.1.0-744ca362/det_ekyc_v1.5.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.1.0-744ca362/rec_ekyc_v1.1.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.3.0': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.1.0-744ca362/det_ekyc_v1.5.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.3.0-e2fe0db2/rec_ekyc_v1.3.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.4.4': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.1.0-744ca362/det_ekyc_v1.5.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.4.4-a0d01e90/rec_ekyc_v1.4.4.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.4.5': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.4.5-bc2cb7f0/det_ekyc_v1.8.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.4.5-bc2cb7f0/rec_ekyc_v1.4.5.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.4.8': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.4.5-bc2cb7f0/det_ekyc_v1.8.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.4.8-eaa6b757/rec_ekyc_v1.4.8.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.5.0': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.5.0-efe2fa9d/det_ekyc_v2.0.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.5.0-efe2fa9d/rec_ekyc_v1.5.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.6.0': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/det_ekyc_v2.2.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/rec_ekyc_v1.6.0.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            },
        },
        'det_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/det_ekyc_v2.2.0_onnx.tar',
            },
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/rec_ekyc_v1.6.0_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        },
    },
    'ekyc.1.7.4': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/det_ekyc_v2.2.0.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.4-7023e812/rec_ekyc_v1.7.4.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            },
        },
        'det_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.6.0-201827f9/det_ekyc_v2.2.0_onnx.tar',
            },
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.4-7023e812/rec_ekyc_v1.7.4_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.7.5': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/rec_ekyc_v1.7.5.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            },
        },
        'det_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3_onnx.tar',
            },
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/rec_ekyc_v1.7.5_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.7.6': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.6-8d431650/rec_ekyc_v1.7.6.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            },
        },
        'det_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3_onnx.tar',
            },
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.6-8d431650/rec_ekyc_v1.7.6_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
    'ekyc.1.8.1': {
        'det': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3.tar',
            },
        },
        'rec': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.8.1-b24ce325/rec_ekyc_v1.8.1.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            },
        },
        'det_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.7.5-3995ff73/det_ekyc_v2.5.3_onnx.tar',
            },
        },
        'rec_onnx': {
            'vi': {
                'url': 'http://minio.dev.ftech.ai/paddleocr-model-ekyc.v1.8.1-b24ce325/rec_ekyc_v1.8.1_onnx.tar',
                'dict_path': './ppocr/utils/dict/vietnamese_dict.txt'
            }
        }
    },
}


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='vi')
    parser.add_argument("--det", type=str2bool, default=False)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument("--module", type=str, default="fschool")
    parser.add_argument("--version", type=str, default="")

    for action in parser._actions:
        if action.dest in ['table_char_dict_path']:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    assert lang in MODEL_URLS[DEFAULT_MODEL_VERSION]['rec'], \
        'param lang must in {}, but got {}'.format(MODEL_URLS[DEFAULT_MODEL_VERSION]['rec'].keys(), lang)
    det_lang = lang
    return lang, det_lang


def get_model_config(version, model_type, lang, use_onnx=False):
    if use_onnx:
        model_type = "_".join([model_type, "onnx"])

    if version not in MODEL_URLS:
        logger.warning('version {} not in {}, use version {} instead'.format(
            version, MODEL_URLS.keys(), DEFAULT_MODEL_VERSION))
        version = DEFAULT_MODEL_VERSION
    if model_type not in MODEL_URLS[version]:
        if model_type in MODEL_URLS[DEFAULT_MODEL_VERSION]:
            logger.warning(
                'version {} not support {} models, use version {} instead'.
                format(version, model_type, DEFAULT_MODEL_VERSION))
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, MODEL_URLS[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)
    if lang not in MODEL_URLS[version][model_type]:
        if lang in MODEL_URLS[DEFAULT_MODEL_VERSION][model_type]:
            logger.warning('lang {} is not support in {}, use {} instead'.
                           format(lang, version, DEFAULT_MODEL_VERSION))
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, MODEL_URLS[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return MODEL_URLS[version][model_type][lang]


def get_model_config_rec(version, rec_url, dict_path, use_onnx=False):
    if (rec_url is None and version is None) or version not in MODEL_URLS.keys():
        rec_url = MODEL_URLS[DEFAULT_MODEL_VERSION]["rec"]["vi"]["url"]
    else:
        # rec_url have higher priority than version
        if rec_url:
            rec_url = rec_url
        else:
            if use_onnx:
                rec_url = MODEL_URLS[version]["rec_onnx"]["vi"]["url"]
            else:
                rec_url = MODEL_URLS[version]["rec"]["vi"]["url"]

    config_list = {
        'url': rec_url,
        'dict_path': dict_path
    }
    return config_list


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)

        if 'config_path' in kwargs.keys():
            cfgs = yaml.load(open(kwargs['config_path'], 'rb'), Loader=yaml.Loader)
            if "Global" in cfgs.keys():
                params.__dict__.update(cfgs['Global'])
            if "Resources" in cfgs.keys():
                params.__dict__.update(cfgs['Resources'])
        else:
            params.__dict__.update(**kwargs)

            # Wrapper variables
            if params.device.lower() == 'cuda' or params.use_gpu:
                params.use_gpu = True
            else:
                params.use_gpu = False
            params.benchmark = params.benchmark or params.logging_time

        if params.version == "":
            module = params.module.lower()
            if module == "fschool":
                params.version = ".".join([module, FSCHOOL_DEFAULT_MODEL_VERSION])
            if module == "ekyc":
                params.version = ".".join([module, EKYC_DEFAULT_MODEL_VERSION])
        else:
            params.version = ".".join([params.version, params.module])

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = params.lang, params.lang

        # init model dir
        if params.det:
            det_model_config = get_model_config(params.version, 'det', det_lang, params.use_onnx)
            params.det_model_dir, det_url = confirm_model_dir_url(
                params.det_model_dir,
                os.path.join(BASE_DIR, 'det', params.version),
                det_model_config['url'])
            maybe_download(params.det_model_dir, det_url, use_onnx=params.use_onnx)

        if params.rec:
            rec_model_config = get_model_config_rec(params.version, params.rec_url, params.rec_char_dict_path, params.use_onnx)
            params.rec_model_dir, rec_url = confirm_model_dir_url(
                params.rec_model_dir,
                os.path.join(BASE_DIR, 'rec', params.version),
                rec_model_config['url'])
            maybe_download(params.rec_model_dir, rec_url, use_onnx=params.use_onnx)

            # if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        if params.use_angle_cls:
            cls_model_config = get_model_config(params.version, 'cls', 'ch')
            params.cls_model_dir, cls_url = confirm_model_dir_url(
                params.cls_model_dir,
                os.path.join(BASE_DIR, 'cls', params.version),
                cls_model_config['url'])
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        print(params)
        # init det_model and rec_model
        super().__init__(params)

    def ocr(self, img, det=False, rec=True, cls=False):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img, cls)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


class PPStructure(OCRSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config(params.version, 'det', det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config(params.version, 'rec', lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'rec', lang),
            rec_model_config['url'])
        table_model_config = get_model_config(params.version, 'table', 'en')
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, VERSION, 'ocr', 'table'),
            table_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(
                Path(__file__).parent / table_model_config['dict_path'])

        print(params)
        super().__init__(params)

    def __call__(self, img):
        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res = super().__call__(img)
        return res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**(args.__dict__))
    elif args.type == 'structure':
        engine = PPStructure(**(args.__dict__))
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
            if result is not None:
                for line in result:
                    logger.info(line)
        elif args.type == 'structure':
            result = engine(img_path)
            save_structure_res(result, args.output, img_name)

            for item in result:
                item.pop('img')
                logger.info(item)
