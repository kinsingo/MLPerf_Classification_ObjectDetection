import os
import subprocess
from python.main import start
import argparse
import shlex

# main.py 파일에서 함수로 변환
def callMain(profile, mlperf_conf, model, dataset_path, output, accuracy):
    print(f"Profile: {profile}")
    print(f"MLPerf Config: {mlperf_conf}")
    print(f"model: {model}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output}")
    print(f"Accuracy Mode: {accuracy}")
    start(profile, mlperf_conf, model, dataset_path, output, accuracy)

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
    os.environ['DATA_DIR'] = os.path.join(os.getcwd(), "real_imagenet")#os.environ['DATA_DIR'] = os.path.join(os.getcwd(), "fake_imagenet")

def Set_Evaluation_Parameters():

    # 1. Bash 스크립트 실행 (run_local.sh)
    bash_script = "./run_local.sh onnxruntime mobilenet cpu --accuracy" 
    #bash_script = "./run_local.sh dxrt mobilenet cpu --accuracy "  
    #bash_script = "./run_local.sh dxrt resnet50 cpu --accuracy "  
    result = subprocess.run(bash_script, shell=True, capture_output=True, text=True)

    # 2. 출력된 명령어 (cmd) 확인
    cmd = result.stdout.strip().splitlines()[-1]  # 마지막 줄을 가져옴
    print(f"Received command from Bash script: {cmd}")

    # 이제 python3 python/main.py 부분을 제외하고 인수를 추출하여 함수 호출

    # 명령어를 shlex를 사용해 토큰화 (인수 분리)
    args = shlex.split(cmd)

    # 'python3 python/main.py' 제외하고 나머지 인수만 추출
    args = args[2:]  # 'python3', 'python/main.py'를 제외

    # 3. 명령줄 인수로 받은 값을 함수 인자로 변환
    arg_dict = {}
    for i in range(0, len(args), 2):
        key = args[i].lstrip("--")
        value = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True
        arg_dict[key] = value
    
    # key-value 쌍 출력
    print("Arguments received:")
    for key, value in arg_dict.items():
        print(f"{key}: {value}")

    callMain(
        profile=arg_dict.get('profile'),
        mlperf_conf=arg_dict.get('mlperf_conf'),
        model=arg_dict.get('model'),
        dataset_path=arg_dict.get('dataset-path'),
        output=arg_dict.get('output'),
        accuracy=arg_dict.get('accuracy', False)
    )

    # 3. 받은 명령어를 실제로 실행할 경우
    # subprocess.run(cmd, shell=True)  # 주석을 해제하면 명령어를 실행

    #try:
    #    # run 명령어를 사용하면 명령어가 성공적으로 실행되었는지 체크하고 결과를 가져올 수 있습니다.
    #    result = subprocess.run(command, check=True, text=True, capture_output=True)
    #    print(result.stdout)  # 성공 시 출력 결과 표시
    #except subprocess.CalledProcessError as e:
    #    print(f"Error occurred: {e.stderr}")  # 오류 시 에러 메시지 표시

StartFromThisDirectory()
Set_Model_And_Dataset_Directories()
Set_Evaluation_Parameters()



