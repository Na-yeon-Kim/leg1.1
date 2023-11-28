#필요한 라이브러리 import
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

app = Flask(__name__)

#모델 예측 함수 정의
def predict_image(image_path):
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 모델 load
    loaded_model = models.resnet34(pretrained=False)
    num_features = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_features, 24)
    loaded_model.load_state_dict(torch.load('trained_model.pth'))
    loaded_model.eval()

    # 이미지 열기 및, transforms 함수를 이용한 전처리
    image = Image.open(image_path)
    image_tensor = transforms_test(image).unsqueeze(0)

    # 모델 예측
    with torch.no_grad():
        outputs = loaded_model(image_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds].item()

    # 결과 반환
    return preds.item(), confidence

# CSS 파일 제공을 위한 라우트
@app.route('/<page>', methods=['GET', 'POST'])
def page(page):
    print("page=", page)
    if ".css":
        return send_from_directory("templates",page)
        return

# 이미지 업로드 및 예측을 처리하는 메인 라우트
@app.route('/', methods=['GET', 'POST'])
def index():

    # 이미지 파일이 POST 요청으로 제출되었는지 확인
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # 디렉토리 생성 및 업로드된 파일 저장
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')

            file_path = 'static/uploads/uploaded_image.jpg'
            file.save(file_path)

            
            try:
                #정의된 모델 예측 함수를 이용해 결과 값 반환
                prediction, confidence = predict_image(file_path)

                confidence_threshold = 0.5

                # 예측 신뢰도에 따라 결과를 표시
                if confidence >= confidence_threshold:
                    class_names =  ['separatory Funnel', 'sieves', '데시케이터', '마이크로피펫', '막자사발', '메스실린더', '부피플라스크', '비커', '삼각플라스크', '속슬렛추출장치', '스포이드 공병', '실험실 클램프', '알코올 램프', '약수저', '여과지', '오실로스코프', '웨잉 디쉬', '임호프콘', '증발 접시', '집기병', '코니칼튜브', '큐벳', '파라필름', '홈판']
                    result = class_names[prediction]
                    return render_template('index.html', result=result, image_path=file_path)
                else:
                    return render_template('index.html', result="잘못된 사진 혹은 아직 업데이트 되지 않은 실험도구인 것 같습니다. 다시 시도해 주세요.", image_path=file_path)
            except RuntimeError as e:
                return render_template('index.html', result="잘못된 사진 혹은 아직 업데이트 되지 않은 실험도구인 것 같습니다. 다시 시도해 주세요.", image_path=file_path)

    # 이미지가 업로드되지 않았을 때 메인 페이지 렌더링
    return render_template('index.html', result=None, image_path=None)

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True)