https://github.com/dbgmlek/yolov4_object_detection_workspace.git
https://paperswithcode.com/ # 논문사이트 

1. conda create -n yolo pip python=3.7
1. conda activate yolo
1. library 설치
    
	- pip install tensorflow <-- 2.4.* 
		- pip show tensorflow 

	- pip install opencv-contrib-python==4.4.*
		-pip show opencv-contr ib-python 
		- 제거 : pip uninstall opencv-contrib-python
        - **opencv 4.4대 버전 설치**

	- pip install easydict
	- pip install pillow

1. clone
	- git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
1. weights 다운로드
	- https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
	- tensorflow-yolov4-tflite/data 폴더 아래 복사

1. 검출 할 class 목록 파일(.names) 작성
1. core/config.py 에 class 목록파일 경로 설정
    - __C.YOLO.CLASSES 변수에 설정
1. weights(가중치)를 Tensorflow 버전으로 변환	
	- `python save_model.py --weights ./weights/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416`
	- python save_model.py --weights ./weights/yolov4_best.weights --output ./checkpoints/yolov4-lt --input_size 416
        - weights: yolo에서 학습한 가중치
        - output: 변환 모델을 저장할 디렉토리. 변환 후 saved model 형식으로 생성됨
        - input_size: input image size
	
	
# Detection

## 이미지
```
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg
```

## 비디오

```
python detectvideo.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/road.mp4
```
