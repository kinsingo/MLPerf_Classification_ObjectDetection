"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import python.backend as backend
from dx_engine import InferenceEngine

class BackendDXRT(backend.Backend):
    def __init__(self):
        super(BackendDXRT, self).__init__()

    def version(self):
        return "0.0.1"

    def name(self):
        return "dxrt"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.ie = InferenceEngine(model_path)
        self.inputs = inputs
        self.outputs = outputs
        return self

    def predict(self, feed):
        feed = feed["Im2col_input"]
        ie_output = self.ie.run(feed)
        return ie_output #여기를 어떻게 반환 하느냐에 따라 Postprocessing 이 달라지는듯 ? dataset.PostProcessArgMax(offset=0) 으로 해도 되는거 같음 (Pytorch 쪽과 비슷)
