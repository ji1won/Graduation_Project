from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import pymysql
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import winsound
import plotly.graph_objs as go
import plotly.express as px


app = Flask(__name__)

socketio = SocketIO(app)

db = pymysql.connect(host='127.0.0.1', 
                     port=3306, user='root', 
                     passwd='qwd1245910!', 
                     db='posedb', 
                     charset='utf8')
cursor = db.cursor()
sql = "INSERT INTO poseTable (pose, time) VALUES (%s, %s)"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


#포즈 라벨 매핑
poses_mapping = {0: 'chin', 
                 1: 'headdown', 
                 2: 'right_pose', 
                 3: 'tilted', 
                 4: 'turtleneck'}

#자세별 경고메세지 매핑
warning_messages = {
    'chin': '턱을 괴고 있습니다.',
    'headdown': '머리를 조금 들어주세요.',
    'tilted': '고개가 기울어졌습니다.',
    'turtleneck': '거북목입니다.'
}

pose_result = ""
warning_message = ""
headdownList = []


def generate():
    global pose_result, warning_message, headdownList
    
    while True:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Null.Frames")
                    break
                if success:
                    # 키포인트를 저장할 리스트 생성
                    sequences = []
                    
                    # 이미지를 RGB로 변환
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 이미지에서 키포인트 추출
                    keypoints = pose.process(image)
                    
                    # 추출된 키포인트를 변수에 저장
                    pose_landmarks = keypoints.pose_landmarks
                    
                    # 이미지에 키포인트 시각화
                    output_frame = image.copy()
                    mp_drawing.draw_landmarks(image=output_frame, landmark_list=pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
                    
                    # 시각화된 이미지를 다시 BGR로 변환
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    
                    # 키포인트 좌표를 리스트로 변환하고, 크기 조정
                    pose_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks.landmark]
                    frame_height, frame_width = output_frame.shape[:2]
                    pose_landmarks *= np.array([frame_height, frame_height, frame_width])
                    
                    # 좌표를 반올림하고 1차원 벡터로 변환하여 저장
                    pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.float32).tolist()
                    
                    # 키포인트 리스트를 sequences에 추가
                    sequences.append(pose_landmarks)
                    
                    # sequences를 numpy 배열로 변환
                    X = np.array(sequences)

                    # 사전에 저장된 모델 불러오기
                    model = tf.keras.models.load_model("C:\\Users\\USER\\Desktop\\flask_opencv\\model\\DNN_3dense_300.h5")
                    
                    # 모델에 입력하여 자세 결과 예측
                    result = model.predict(X)

                    # 결과에서 가장 높은 확률의 자세 레이블을 가져옴
                    pose_result = poses_mapping[np.argmax(result[0])]


                    # right_pose가 아닐 경우 해당 맞춤 경고메세지 표시 
                    if pose_result != 'right_pose':
                        warning_message = warning_messages[pose_result]
                    else:
                        warning_message = ""

                    # db에 sql문 실행하여 pose에 자세라벨, time에 시간 저장하기
                    cursor.execute(sql, (poses_mapping[np.argmax(result[0])], datetime.now().strftime('%Y.%m.%d - %H:%M:%S')))
                    db.commit()


                    # headdown 자세가 10번 연속될 경우 알람음+알림창
                    if poses_mapping[np.argmax(result[0])] == 'headdown':
                         headdownList.append(1)
                         if len(headdownList) == 10 : #리스트 안에 10개 값이 차면 경고음 출력.
                              # 알림음 재생
                              winsound.Beep(
                              frequency=440, 
                              duration=1000 
                              )
                              socketio.emit('headdown_alert', '고개를 들어주세요')
                              headdownList = []
                    else:
                        headdownList = []  # 다른 자세가 나타나면 리스트 초기화


                    
                    ret, buffer = cv2.imencode('.jpg', output_frame)
                    frame = buffer.tobytes()
                    cv2.waitKey(1000)

                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pose_result")
def get_pose_result():
    global pose_result
    return jsonify(result=pose_result)

@app.route("/warning_message")
def get_warning_message():
    global warning_message
    return jsonify(message=warning_message)

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/posedb')
def posedb():

    # 자세별 비율 pie 차트 표시 (pie_html)
    # SQL문 작성
    sql2 = """
        SELECT pose, COUNT(*) as count
        FROM poseTable
        GROUP BY pose
        ORDER BY count DESC
    """
    # SQL문 실행
    cursor = db.cursor()
    cursor.execute(sql2)
    # 결과 가져오기
    results = cursor.fetchall()

    # 결과를 백분율로 변환하기
    total_count = sum(row[1] for row in results)
    percentages = [(row[0], row[1] / total_count * 100) for row in results]

    # 데이터 준비
    pie_labels = [row[0] for row in percentages]
    pie_percentages = [row[1] for row in percentages]

    # 시각화
    pie_data = {'labels': pie_labels, 'values': pie_percentages}
    pie_fig = px.pie(pie_data, values='values', names='labels')

    pie_html = pie_fig.to_html(full_html=False)



    # weekusage_html 그래프
    # poseTable에서 데이터 가져오기
    cursor.execute("""
        SELECT DATE(time) AS date, pose
        FROM poseTable
        WHERE time >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
    """)
    results = cursor.fetchall()
    data = pd.DataFrame(results, columns=['Date', 'Pose'])

    # 일별 누적 사용시간 계산
    data['UsageTime'] = 1
    week_data = data.groupby(['Date', 'Pose']).sum().groupby(level=[1]).cumsum().reset_index()

    # 초 단위 사용시간을 분으로 환산
    week_data['UsageTime'] = week_data['UsageTime'].apply(lambda x: x / 60)

    # 히스토그램 그래프 생성
    poses = week_data['Pose'].unique()
    fig = go.Figure()
    for pose in poses:
        pose_data = week_data[week_data['Pose'] == pose]
        fig.add_trace(go.Bar(x=pose_data['Date'], y=pose_data['UsageTime'], name=pose))

    fig.update_layout(xaxis_title='날짜', yaxis_title='누적 사용시간 (분)')

    # 각 라벨 개수를 초로 계산하여 분 단위로 환산하여 표시
    for data in fig.data:
        #pose_label = data.name
        pose_count = data.y
        pose_time = pose_count * 1 / 60
        data.text = [f'{t:.2f}분' for t in pose_time]

    weekusage_html = fig.to_html(full_html=False)



    #dayusage_html 그래프
    # poseTable에서 데이터 가져오기
    cursor.execute("""
        SELECT DATE(time) AS date, pose
        FROM poseTable
        WHERE time >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
    """)
    results = cursor.fetchall()
    data = pd.DataFrame(results, columns=['Date', 'Pose'])

    # 날짜별 사용시간 계산
    data['UsageTime'] = 1
    day_data = data.groupby(['Date']).sum().reset_index()

    # 초 단위 사용시간을 분으로 환산
    day_data['UsageTime'] = day_data['UsageTime'].apply(lambda x: x / 60)

    # 히스토그램 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Bar(x=day_data['Date'], y=day_data['UsageTime']))

    fig.update_layout(xaxis_title='날짜', yaxis_title='누적 사용시간 (분)')

    # 그래프를 HTML로 변환하여 dayusage_html에 저장합니다.
    dayusage_html = fig.to_html(full_html=False)


    return render_template('posedb.html', pie_html=pie_html, weekusage_html=weekusage_html, dayusage_html=dayusage_html)



@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app, debug=False)
    cap.release()
    db.close()
    