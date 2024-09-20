#!/bin/bash

# 1. 공통 설정 스크립트 로드
# 이 스크립트는 공통 설정 또는 함수들을 포함하고 있는 'run_common.sh' 스크립트를 소스로 가져옵니다.
# 'run_common.sh' 파일에는 이 스크립트에서 사용할 환경 변수나 함수들이 정의되어 있을 가능성이 있습니다.
source ./run_common.sh

# 2. 공통 옵션 설정
# 'mlperf.conf' 파일의 경로를 포함하는 옵션을 설정합니다.
common_opt="--mlperf_conf ../../mlperf.conf"

# 3. 데이터셋 경로 설정
# 환경 변수 $DATA_DIR을 사용하여 데이터셋의 경로를 설정합니다.
dataset="--dataset-path $DATA_DIR"

# 4. 출력 디렉토리 설정
# 출력 디렉토리가 설정되어 있지 않으면, 현재 디렉토리 하위의 'output/$name' 디렉토리를 기본값으로 사용합니다.
OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/output/$name}

# 5. 출력 디렉토리가 존재하지 않으면 생성
# 지정된 출력 디렉토리가 존재하지 않을 경우, 해당 디렉토리를 생성합니다.
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 6. 명령행 인수 처리 (공백 또는 작은따옴표 포함)
# 명령행 인수에 공백(' ') 또는 작은따옴표(')가 포함된 경우 이를 처리합니다.
# '$1'은 명령행 인수의 첫 번째 값을 의미하며, 공백 또는 작은따옴표가 포함된 경우 추가 따옴표로 감쌉니다.
# 그런 다음, 'ARGS' 변수에 모든 인수를 하나의 문자열로 저장합니다.
pattern=" |'" # 공백 또는 작은따옴표 패턴
while [ -n "$1" ]; do
    # 인수에 패턴이 포함되어 있으면 따옴표로 감싸서 추가
    if [[ $1 =~ $pattern ]]; then
        ARGS=$ARGS' "'$1'"'
    else
        # 그렇지 않으면 그대로 추가
        ARGS="$ARGS $1"
    fi
    shift  # 다음 인수로 이동
done

# 7. 명령어 작성 및 실행
# Python 스크립트를 실행하기 위한 명령어를 구성합니다.
# 명령어는 Python 스크립트와 각종 설정을 인수로 포함하며, Python 3을 사용하여 실행됩니다.
# `eval`을 사용하여 구성된 명령어를 실행합니다.
cmd="python3 python/main.py --profile $profile $common_opt --model \"$model_path\" $dataset \
    --output \"$OUTPUT_DIR\" $EXTRA_OPS ${ARGS}"

# 8. 구성된 명령어를 출력하고 실행
# 최종적으로 구성된 명령어를 echo로 출력한 후, `eval`을 사용해 실행합니다.
echo $cmd
eval $cmd
