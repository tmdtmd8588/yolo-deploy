# 1. Python 3.11 slim 이미지 기반
FROM python:3.11-slim

# 2. 컨테이너 안에서 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 코드 복사
COPY . .

# 5. 컨테이너에서 열어줄 포트
EXPOSE 8000

# 6. 실행 명령 (FastAPI + Uvicorn 실행)
CMD ["uvicorn", "main_yolo2:app", "--host", "0.0.0.0", "--port", "8000"]
