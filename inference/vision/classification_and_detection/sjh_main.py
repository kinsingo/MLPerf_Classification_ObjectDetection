import os
import subprocess

# 실제 실행되는 함수
# cmd="python3 python/main.py --profile $profile $common_opt --model \"$model_path\" $dataset \
#     --output \"$OUTPUT_DIR\" $EXTRA_OPS ${ARGS}"


def StartFromThisDirectory():
    directory = "/home/orangepi/mlperf-classification-sjh/inference/vision/classification_and_detection/"
    try:
        # 디렉토리 변경
        os.chdir(directory)
        print(f"Current working directory changed to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

def Set_Model_And_Dataset_Directories():
    print(f'os.getcwd() : {os.getcwd()}')
    os.environ['MODEL_DIR'] = os.path.join(os.getcwd(), "models")
    os.environ['DATA_DIR'] = os.path.join(os.getcwd(), "fake_imagenet")

def Set_Evaluation_Parameters():

    # run_common.sh 에서 Setting 한 부분
    # if [ $name == "mobilenet-onnxruntime" ] ; then
    #     model_path="$MODEL_DIR/mobilenet_v1_1.0_224.onnx"
    #     profile=mobilenet-onnxruntime
    command = ["./run_local.sh", "onnxruntime", "mobilenet", "cpu", "--accuracy"]     #!./run_local.sh onnxruntime mobilenet cpu --accuracy 

    try:
        # run 명령어를 사용하면 명령어가 성공적으로 실행되었는지 체크하고 결과를 가져올 수 있습니다.
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)  # 성공 시 출력 결과 표시
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")  # 오류 시 에러 메시지 표시

StartFromThisDirectory()
Set_Model_And_Dataset_Directories()
Set_Evaluation_Parameters()



