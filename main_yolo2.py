import threading # 사람 탐지 로직을 백그라운드에서 실행
import cv2 # OpenCV, 비디오 프레임 읽고 시각화에 사용
import warnings 
from fastapi import FastAPI # REST API 서버 프레임워크
import uvicorn # FastAPI 앱을 실행하기 위한 ASGI 서버
from ultralytics import YOLO # YOLOv8 객체 탐지 모델
from fastapi.middleware.cors import CORSMiddleware 
import time
import math

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI() # FastAPI 객체 생성

app.add_middleware( # CORS (Cross-Origin Resource Sharing) 문제 해결을 위한 미들웨어 추가, 클라이언트가 다른 도메인에서 이 API에 요청할 수 있도록 허용
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 요청 허용 (보안상 실 서비스에선 특정 도메인만 허용하는 것이 좋음) ["http://localhost:포트번호"]로 제한 가능
    allow_credentials=True, # 쿠키, 인증 정보 등을 포함한 요청 허용
    allow_methods=["*"], # GET, POST, PUT 등 모든 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# 카메라별 사람 수 저장
camera_counts = [0, 0]  # [카메라1, 카메라2]
wait_time = 0 # 예상 대기 시간 (분)
average_counts = [] # 평균 계산용 리스트

PERSON_CLASS_ID = 0 # YOLOv8 모델에서 ID: 0번이 사람
count_lock = threading.Lock() # threading.Lock 사용

model = YOLO("yolov8n.pt") # YOLOv8 모델 로드

video_paths = [ # 감지할 비디오 파일 경로
    "http://172.30.1.42:8080/video",  # 카메라1
    "http://172.30.1.29:8080/video",  # 카메라2
]

def detect_people(camera_index, video_path): # 사람 탐지 함수
    global camera_counts 
    cap = cv2.VideoCapture(video_path) # OpenCV의 VideoCapture 객체를 생성

    while True: # 무한 루프 시작
        ret, frame = cap.read()  # cap.read()로 영상에서 프레임을 하나씩 읽음
        if not ret: # ret == False이면 루프 종료
            break

        frame = cv2.resize(frame, (640, 360)) # YOLO 처리 속도 향상을 위해 프레임을 640x360으로 리사이즈
        results = model(frame, conf=0.2, iou=0.5, max_det=20, verbose=False)[0]  # 프레임을 YOLO 모델에 입력하고, 첫 번째 결과 ([0])를 가져옴
        #하이퍼파라미터 #conf(기본 0.25, 낮추면 더 많이 탐지하지만 오탐 증가) # iou(기본 0.7, 낮추면 중복 제거 강하게 적용됨) # max_det(한 프레임에서 최대 탐지 수)

        person_detections = [box for box in results.boxes if int(box.cls[0]) == PERSON_CLASS_ID]

        with count_lock: # 사람 수 변경 시
            camera_counts[camera_index] = len(person_detections)
            current_wait_time = wait_time #임시로 예상대기시간도 표시하기위해 추가

        # 디스플레이 (카메라별 개별 창)
        for box in person_detections: # 박스 그리기
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # 경계 상자 좌표
            conf = float(box.conf[0]) # 신뢰도(확률)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 프레임에 초록색 박스를 그림
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), # 프레임에 라벨을 그림
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        total_count = sum(camera_counts)
        
        cv2.putText(frame, f'Cam{camera_index+1}: {camera_counts[camera_index]} | Total: {total_count}', (10, 30), # 현재 감지된 사람 수를 좌상단에 표시
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 예상 대기 시간 표시
        cv2.putText(frame, f'Wait Time: {current_wait_time} min', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        #배포를 위해 주석처리
        #cv2.imshow(f'Camera {camera_index+1}', frame) # 프레임을 실시간으로 화면에 표시
        #if cv2.waitKey(1) & 0xFF == 27: # 사용자가 ESC키를 누르면 루프 종료
            #break

    cap.release() # cap 객체가 사용하던 영상 스트림을 종료
    #cv2.destroyAllWindows() # OpenCV가 생성한 모든 창(윈도우)을 닫음

def calculate_wait_time(): #예상대기시간을 구하는 함수
    global wait_time
    while True:
        for _ in range(6):
            with count_lock:
                total_people = sum(camera_counts)
                if total_people >= 3: # 만약 사람이 3명 이상이면 -3을 한다
                    average_counts.append(total_people - 3)
                else:
                    average_counts.append(total_people)
            time.sleep(10)

        avg = sum(average_counts) / len(average_counts)

        wait_time = round(avg*20/60)  # 1명당 20초 # 초를 분으로 바꾸고 반올림함 

        average_counts.clear()

@app.get("/api/estimate/lilac") # FastAPI의 데코레이터로, "/api/estimate/lilac" 경로에 대해 GET 요청을 처리하도록 지정
def get_lilac(): # API 요청이 들어올 때 실행될 핸들러 함수 # API 호출 시
    with count_lock:
        return {"cam1": camera_counts[0], "cam2": camera_counts[1],
                "total": sum(camera_counts), "wait_time": wait_time}

@app.get("/api/estimate/dalelac/korea") 
def get_lilac(): 
    with count_lock:
        return {"cam1": camera_counts[0], "cam2": camera_counts[1],
                "total": sum(camera_counts), "wait_time": wait_time}
    
@app.get("/api/estimate/dalelac/japan") 
def get_lilac(): 
    with count_lock:
        return {"cam1": camera_counts[0], "cam2": camera_counts[1],
                "total": sum(camera_counts), "wait_time": wait_time}
    
@app.get("/api/estimate/dalelac/specialty")
def get_lilac(): 
    with count_lock:
        return {"cam1": camera_counts[0], "cam2": camera_counts[1],
                "total": sum(camera_counts), "wait_time": wait_time}
    
@app.on_event("startup") # FastAPI 서버가 실행될 때 한 번 실행되는 이벤트
def startup_event():
    # 카메라 2대 각각 스레드 실행
    for idx, path in enumerate(video_paths):
        threading.Thread(target=detect_people, args=(idx, path), daemon=True).start()
    threading.Thread(target=calculate_wait_time, daemon=True).start()

if __name__ == "__main__": # 현재 스크립트가 직접 실행될 때만 내부 코드를 실행
    uvicorn.run("main_yolo2:app", host="0.0.0.0", port=8000) # FastAPI 서버를 실행하는 명령



