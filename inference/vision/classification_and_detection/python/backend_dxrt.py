"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import backend
from dx_engine import InferenceEngine

class BackendDXRT(backend.Backend):
    def __init__(self):
        super(BackendDXRT, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        return "dxrt"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.ie = InferenceEngine(model_path)
        self.inputs = inputs
        self.outputs = outputs
        return self

    def predict(self, feed):
        ie_output = self.ie.run(feed)
        return ie_output #여기를 어떻게 반환 하느냐에 따라 Postprocessing 이 달라지는듯 ? dataset.PostProcessArgMax(offset=0) 으로 해도 되는거 같음 (Pytorch 쪽과 비슷)
