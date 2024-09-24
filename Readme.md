# Progress
 - To-Do (as of 20.09.20)
 1. main.py 쪽 보고 하기 3가지 구현 다 끝낸거 같음, 이제 테스트 해보자 !! 
    1) pre-processing
    2) Inference
    3) post-processing

- 24.09.19
 1. Repo 생성 (Classfication 이랑 ObjectDetection 만 저장)
 2. OrangePi 를 SSH 통해서 Window 환경에서 접근 가능하도록 함 (orangepi@147.46.168.115:8024)
 3. dataset 경로는 "home/orangepi/dataset/imageNet2012-validation" 여기임, 각 class 별로 하나의 폴더, 각 폴더는 50 Samples (50000 = 1000 x 50)
 
 - 24.09.20
 1. "imageNet_example.py" 를 수정해서 성공적으로 돌림 (4ea Sample Image 중에 3개 맞춤, 1.jpeg 는 Toast 로 맞추었음 (실제로 헷갈림))
 2.  MLPerf 의 Resnet50.onnx 를 Resnet50.dnnx 로 변환해서 테스트 결과, 결과값이 이상함 
     (Compilation 자체는 성공 했으나, Config 를 좀 수정해야 하나 싶음) 
 3. MLPerf 의 python Interpreter 세팅 진행 (DX-RT가 Default로 다운로드 되었으니(특정 Environment 에 다운로드 된 것이 아님))
 4. 가상 환경은 python 3.8.10 64-bit (/bin/python)  사용 
 5. "inference/vision/classification_and_detection/sjh_main.py" 파일 실행하는거로 평가 예정. 

 6. run_common.sh 에 아래 내용 추가했음 (dnnx model을 하기 model_path 쪽에 넣어둠, dataset 은 우선 fake_imagenet 폴더 사용해서 평가, 추후 Imagenet 으로 다시 평가)
   if [ $name == "resnet50-dxrt" ] ; then
      model_path="$MODEL_DIR/ResNet50-1.dxnn"
      profile=resnet50-dxrt
   fi
   if [ $name == "mobilenet-dxrt" ] ; then
      model_path="$MODEL_DIR/MobileNetV1-1.dxnn"
      profile=mobilenet-dxrt
   fi

7. run_common.sh 를 다음과 같이 수정 (dxrt 추가)
if [ $# -lt 1 ]; then
    echo "usage: $0 dxrt|tf|onnxruntime|pytorch|tflite|tvm-onnx|tvm-pytorch|tvm-tflite [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|retinanet] [cpu|gpu]"
    exit 1
fi
if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

for i in $* ; do
    case $i in
       dxrt|tf|onnxruntime|tflite|pytorch|tvm-onnx|tvm-pytorch|tvm-tflite|ncnn) backend=$i; shift;;
       cpu|gpu|rocm) device=$i; shift;;
       gpu) device=gpu; shift;;
       resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|ssd-resnet34-tf|retinanet) model=$i; shift;;
    esac
done

 8. main.py 의 SUPPORTED_PROFILES 구현 (아래 SUPPORTED_DATASETS 의 imagenet-dxrt-resnet50-mobilenet 을 dataset 으로 불러오도록 함)
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

 9. main.py 의 SUPPORTED_DATASETS 구현 (dataset.pre_process_dxrt 구현 필요)
 SUPPORTED_DATASETS = {
    "imagenet-dxrt-resnet50-mobilenet":
        (imagenet.Imagenet, dataset.pre_process_dxrt, dataset.PostProcessArgMax(offset=0),
         {"image_size": [224, 224, 3]}),
 };

 10. dataset.py 의 pre_process_dxrt 를 하기와 같이 구현
 def pre_process_dxrt(img, dims=None, need_transpose=False):
    new_shape=(224, 224)
    align=64
    format=cv2.COLOR_BGR2RGB
    image = cv2.resize(image, new_shape)
    h, w, c = image.shape
    if format is not None:
        image = cv2.cvtColor(image, format)
    if align == 0 :
        return image
    length = w * c
    align_factor = align - (length - (length & (-align)))
    image = np.reshape(image, (h, w * c))
    dummy = np.full([h, align_factor], 0, dtype=np.uint8)
    image_input = np.concatenate([image, dummy], axis=-1)
    return image_input

11. main 의 get_backend(backend) 함수에서 BackendDxrt 불러 오도록 함 (backend_dxrt.py 의 BackendDXRT 구현 필요)
   def get_backend(backend):
      if backend == "dxrt":
        from backend_dxrt import BackendDXRT
        backend = BackendDxrt()

12. backend_dxrt.py 의 BackendDXRT 하기와 같이 구현함
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


- MLPerf 에 DeepX 올리면서 실제로 어려웠던점
 0. Dataset 다운로드 (약 하루 소요) : 자체적으로 Data Download 가능하도록 WAS 에서 제공하고, Dataset 자체에 대한 설명 및 전처리 하는부분이 있으면 좋을듯함.

 1. MLPerf 를 실행하기 위한 Python Interpreter 환경 설정 (Python Script 를 실행하기 위한 환경설정이 복잡함, C++로 Build 해서 실행하려면 더 복잡할 것으로 보임) 
    : 환경 설정 과정에서, 예상치 못하게 필요 Package 설치가 잘 안되는 경우들이 있어, 이러한 경우 직접 원인 파악 및 설치가 필요함. 

 2. MLPerf 의 어떤 부분을 어떤식으로 수정해야 하는지, Open Source 구조 파악하는데 시간이 필요함 (약 하루 소요)
    : Shell Script 기반으로 Command 로 기존 구현에 대한 평가를 용이하게는 만들었으나, 오히려 MLPerf 구현부를 수정해서 평가하기 위해서는 Shell Script 에 대한 분석도 
    별도로 진행 되어야 하기에 구조적으로 복잡하게 보임. Shell Script 그리고 Python Script 가 어떤식으로 상호 동작하는지 구체적인 설명이 없기에, 직접 하나씩 Script 를 모듈별로 분석 해야함. 
    (Error 발생 시, Shell Script 는 Debuging 이 안되기 때문에, 문제점 찾아내기 어려움)