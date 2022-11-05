import numpy as np

import unittest
from unittest import TestCase
from paddleocr import PaddleOCR


class TestPaddleOCR(TestCase):
    def test_is_string(self):
        ocr = PaddleOCR(use_gpu=False)
        s = ocr.ocr("test_img.png")

        assert len(s) == 1
        assert type(s) == list
        assert type(s[0][0]) == str
        assert type(s[0][1]) == float

    def test_module_for_ekyc(self):
        ocr = PaddleOCR(
            module='ekyc',
            det=True,
        )
        s = ocr.ocr(
            "id_card.jpg",
            det=True,
        )

        self.assertTrue(
            any([
                isinstance(el, list) and
                isinstance(el[0], list) and
                isinstance(el[1], tuple) and
                isinstance(el[1][0], str) and
                isinstance(el[1][1], float)
                for el in s
            ])
        )

    def test_is_string_onnx(self):
        ocr = PaddleOCR(
            use_gpu=False,
            use_onnx=True
        )
        s = ocr.ocr("test_img.png")

        assert len(s) == 1
        assert type(s) == list
        assert type(s[0][0]) == str
        assert type(s[0][1]) == float

    def test_module_for_ekyc_onnx(self):
        ocr = PaddleOCR(
            module='ekyc',
            det=True,
            use_onnx=True
        )
        s = ocr.ocr(
            "id_card.jpg",
            det=True,
        )

        self.assertTrue(
            any([
                isinstance(el, list) and
                isinstance(el[0], list) and
                isinstance(el[1], tuple) and
                isinstance(el[1][0], str) and
                isinstance(el[1][1], float)
                for el in s
            ])
        )


if __name__ == '__main__':
    unittest.main()