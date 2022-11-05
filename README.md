# 📋 Paddle OCR For Vietnamese Text OCR

## :rocket: 1. Quickly use as Library 
```
pip3 install http://minio.dev.ftech.ai/paddleocr-3.10.0-aaecb4e1/paddleocr-3.10.0-py3-none-any.whl
```

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(rec_model_dir='{your_rec_model_dir}')
img_path = '{test_image_file}'
result = ocr.ocr(img_path)

for line in result:
    print(line)
```


## :wrench: 2. Install

###### Prerequisite:
* Python: 3.7
* Virtualenv

###### Requirements:
```bash
python3 -m venv venv
pip3 install --upgrade pip
pip3 install paddlepaddle-gpu
pip3 install -r requirements.txt
````

:sos: :sos: Other CUDA version, refer this [link](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html).

###### Documents:

* 🐙 Original github repo: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* 📑 Paper v1: [PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/pdf/2009.09941.pdf)
* 📑 Paper v2: [PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System](https://arxiv.org/pdf/2109.03144.pdf)
* 🛠 Installation and quick start: [Here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/quickstart_en.md)
* 🛠 Training recognition model on the custom dataset: [Here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/recognition_en.md)
* 📝 Master sheet: [PaddleOCR Model Tracking](https://docs.google.com/spreadsheets/u/1/d/1P_MoePMECAiQd8Bjyv2sd5lcorBlcUhg6aDgR9-lzrg/edit#gid=0)
* 📜 Coda documents: [F-school coda](https://coda.io/d/Fschool-DCU_dI6xjhWtSpN/README_sulfz#_lubLQ)

###### Resource:

* 📄 Training/Testing data: [TextOCR DVC](https://docs.google.com/spreadsheets/d/1uWc8mUXKbr4pI5Z9DVVFb5DpEIcEJwiBPqbVLMiCuZk/edit#gid=1708604277)
* 📄 Model checkpoint: [Google Drive](https://drive.google.com/drive/u/0/folders/11ObNLp-KhKmAJ7Gb3eTDcp1H0XT1sZb5)


###### Table of pre-trained models:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|80.9%|rec_r34_vd_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|78.05%|rec_mv3_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|82.76%|rec_r34_vd_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|79.97%|rec_mv3_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|84.44%|rec_r34_vd_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|rec_mv3_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|MobileNetV3|82.5%|rec_mv3_tps_bilstm_att |[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|RARE|Resnet34_vd|83.6%|rec_r34_vd_tps_bilstm_att |[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 88.52% | rec_r50fpn_vd_none_srn |[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar)|
|NRTR|NRTR_MTB| 84.3% | rec_mtb_nrtr | [Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |

Please refer to the document for training guide and use of PaddleOCR [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/recognition_en.md).

## :guitar: 3. Training Models:

After download dataset and pre-trained model. Run this CMD to training model and evaluate models:

Training:
```bash
python3 tools/train.py -c $PATH_TO_CONFIG_FILE
```
Distributed Training:
```bash
python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c $PATH_TO_CONFIG_FILES
```

Evaluating:
```bash
python3 -m tools/eval.py -c $PATH_TO_CONFIG_FILES -o Global.checkpoints=$PATH_TO_CHECKPOINTS
```

Evaluation with Paddle Inference:

```bash
python3 -m tools/eval_infer_det.py --data_dir $DATA_DIR ----label_file_list $LABEL_FILE_LIST
python3 -m tools/eval_infer_rec.py --data_dir $DATA_DIR ----label_file_list $LABEL_FILE_LIST
```

**_Note:_**

* When training model, You need to update your config files for dataset, checkpoint dir, ...
* To get other configs, for other backbones, metrics, ... Please refer this [link](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.3/configs/rec)

## :musical_keyboard: 4. OCR Prediction Engine:

#### :pushpin: 4.1. Paddle Inference:
* Convert training model to Paddle Inference

    ```bash
    python3 tools/export_model.py -c $CONFIG_PATH -o Global.save_model_dir=$SAVE_DIR_PATH
    ```
  
###### :strawberry: Create PaddleOCR Instances using Paddle Inference:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_gpu=True,
)        
```


#### :pushpin: 4.2. ONNX:

###### :strawberry: Convert model to ONNX:

* Install paddle2onnx

    ```bash
    pip3 install paddle2onnx==0.9.0
    ```    
  
* Convert paddle detection to onnx model
    ```bash
    paddle2onnx --model_dir ./inference/det \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./inference/det_onnx/model.onnx \
    --opset_version 10 \
    --input_shape_dict="{'x':[-1,3,-1,-1]}" \
    --enable_onnx_checker True
    ```
  
* Convert paddle recognition to onnx model
    ```bash
    paddle2onnx --model_dir ./inference/rec \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./inference/rec_onnx/model.onnx \
    --opset_version 10 \
    --input_shape_dict="{'x':[-1,1,32,512]}" \
    --enable_onnx_checker True
    ```
  
###### :strawberry: Create PaddleOCR Instances using ONNX:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_gpu=True,
    use_onnx=True, # Use use_onnx args for inference with onnx
)        
```
  
#### :pushpin: 4.3. TensorRT:

**_Note_**: TensorRT need to be installed first.

###### :whale: Run with Docker Images:

* Run docker image:

    ```bash
    docker pull nvcr.io/nvidia/tensorrt:21.03-py3
    docker run --gpus all -it --rm -v $(LOCAL_VOLUME):$(CONTAINER_VOLUME) nvcr.io/nvidia/tensorrt:21.03-py3
    ```

* Install paddlepaddle support TensorRT:
    ```bash
    pip3 install https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0.post111-cp38-cp38-linux_x86_64.whl
    ```

    :sos: :sos: For others CUDA Version or Python Version, please refer this [link](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)

###### :strawberry: Create PaddleOCR Instances using TensorRT:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_gpu=True,
    use_tensorrt=True,
    precision="fp16" # Use precision args for inference with TensorRT (fp16 or fp32)
 )
 ```
