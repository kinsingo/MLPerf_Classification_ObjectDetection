# Progress
- 24.09.19
 1. Repo 생성 (Classfication 이랑 ObjectDetection 만 저장)
 2. OrangePi 를 SSH 통해서 Window 환경에서 접근 가능하도록 함 (orangepi@147.46.168.115:8024)
 3. dataset 경로는 "home/orangepi/dataset/imageNet2012-validation" 여기임, 각 class 별로 하나의 폴더, 각 폴더는 50 Samples (50000 = 1000 x 50)
 
 - 24.09.20
 1. "imageNet_example.py" 를 수정해서 성공적으로 돌림 (4ea Sample Image 중에 3개 맞춤, 1.jpeg 는 Toast 로 맞추었음 (실제로 헷갈림))
 2.  MLPerf 의 Resnet50.onnx 를 Resnet50.dnnx 로 변환해서 테스트 결과, 결과값이 이상함 
     (Compilation 자체는 성공 했으나, Config 를 좀 수정해야 하나 싶음) 

 - To-Do
 1. MLPerf 의 python Interpreter 세팅 진행 (DX-RT가 Default로 다운로드 되었으니(특정 Environment 에 다운로드 된 것이 아님))
 2. 가상 환경은 [/bin/python 에 있는 python3.8.10 64-bit 사용]
