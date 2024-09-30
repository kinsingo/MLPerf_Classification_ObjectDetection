# 중요한것
: 하기 3개 구현 하면 됨
1. pre-processing
2. Inference
3. post-processing

# 실제 구현 과정
 1. "imageNet_example.py" 를 수정해서 성공적으로 돌림 (4ea Sample Image 중에 3개 맞춤, 1.jpeg 는 Toast 로 맞추었음 (실제로 헷갈림))
 2.  MLPerf 의 Resnet50.onnx 를 Resnet50.dnnx 로 변환해서 테스트 결과, 결과값이 이상함 
     (Compilation 자체는 성공 했으나, Config 를 좀 수정해야 하나 싶음)
 4. MLPerf 의 python Interpreter 세팅 진행 (DX-RT가 Default로 다운로드 되었으니(특정 Environment 에 다운로드 된 것이 아님))
 5. 가상 환경은 python 3.8.10 64-bit (/bin/python)  사용 
 6. "inference/vision/classification_and_detection/sjh_main.py" 파일 실행하는거로 평가 예정. 

 7. run_common.sh 에 아래 내용 추가했음 (dnnx model을 하기 model_path 쪽에 넣어둠, dataset 은 우선 fake_imagenet 폴더 사용해서 평가, 추후 Imagenet 으로 다시 평가)
```bash
   if [ $name == "resnet50-dxrt" ] ; then
      model_path="$MODEL_DIR/ResNet50-1.dxnn"
      profile=resnet50-dxrt
   fi
   if [ $name == "mobilenet-dxrt" ] ; then
      model_path="$MODEL_DIR/MobileNetV1-1.dxnn"
      profile=mobilenet-dxrt
   fi
```

9. run_common.sh 를 다음과 같이 수정 (dxrt 추가)
```bash
if [ $# -lt 1 ]; then
    echo "usage: $0 dxrt|tf|onnxruntime|pytorch|tflite|tvm-onnx|tvm-pytorch|tvm-tflite [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|retinanet] [cpu|gpu]"
    exit 1
fi

for i in $* ; do
    case $i in
       dxrt|tf|onnxruntime|tflite|pytorch|tvm-onnx|tvm-pytorch|tvm-tflite|ncnn) backend=$i; shift;;
       cpu|gpu|rocm) device=$i; shift;;
       gpu) device=gpu; shift;;
       resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|ssd-resnet34-tf|retinanet) model=$i; shift;;
    esac
done
```

 8. main.py 의 SUPPORTED_PROFILES 구현 (아래 SUPPORTED_DATASETS 의 imagenet-dxrt-resnet50-mobilenet 을 dataset 으로 불러오도록 함)
```python
 SUPPORTED_PROFILES = {
     "resnet50-dxrt": {
        "dataset": "imagenet-dxrt-resnet50-mobilenet",
        "outputs": "ArgMax:0",
        "backend": "dxrt",
        "model-name": "resnet50",
    },
        "mobilenet-dxrt": {
        "dataset": "imagenet-dxrt-resnet50-mobilenet",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "backend": "dxrt",
        "model-name": "mobilenet",
    },
 };
```

 10. main.py 의 SUPPORTED_DATASETS 직접 구현하는 [dataset.pre_process_dxrt, dataset.PostProcessDXRT_SingleStream_ArgMax()] 불러옴
```python
SUPPORTED_DATASETS = {
    #resnet50-dxrt, mobilenet-dxrt
    "imagenet-dxrt-resnet50-mobilenet":
        (imagenet.Imagenet, dataset.pre_process_dxrt, dataset.PostProcessDXRT_SingleStream_ArgMax(),
         {"image_size": [224, 224, 3]}),
```

 11. dataset.py 의 pre_process_dxrt, PostProcessDXRT_SingleStream_ArgMax 를 하기와 같이 구현
```python
#resnet50-dxrt, mobilenet-dxrt
def pre_process_dxrt(img, dims=None, need_transpose=False):
    output_height, output_width, _ = dims #224, 224, 3
    align=64
    format=cv2.COLOR_BGR2RGB
    img = cv2.resize(img, (output_width,output_height))
    h, w, c = img.shape
    if format is not None:
        img = cv2.cvtColor(img, format)
    if align == 0 :
        return img
    length = w * c
    align_factor = align - (length - (length & (-align)))
    img = np.reshape(img, (h, w * c))
    dummy = np.full([h, align_factor], 0, dtype=np.uint8)
    image_input = np.concatenate([img, dummy], axis=-1)
    return image_input

class PostProcessDXRT_SingleStream_ArgMax:
    def __init__(self):
        self.good = 0
        self.total = 0

    def __call__(self, results, ids, expected=None, result_dict=None):
        processed_results = []
        result = np.argmax(results[0])
        processed_results.append([result])
        if result == expected[0]:
            self.good += 1
        self.total += 1
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False, output_dir=None):
        results["good"] = self.good
        results["total"] = self.total
```

11. main 의 get_backend(backend) 함수에서 BackendDxrt 불러 오도록 함 (backend_dxrt.py 의 BackendDXRT 구현 필요)
```python
   def get_backend(backend):
      if backend == "dxrt":
        from backend_dxrt import BackendDXRT
        backend = BackendDxrt()
```

12. backend_dxrt.py 의 BackendDXRT 하기와 같이 구현함
```python
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
        return ie_output
```
