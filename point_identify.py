import cv2
import matplotlib.pyplot as plt

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)        # 0 because it is capturing feed from main webcam linked to device

while True:
    ret,frame = video_capture.read()
    image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blob = cv2.dnn.blobFromImage(image=image_grey, scalefactor=1.0 / 255, size=(image_grey.shape[1], image_grey.shape[0]))

    network = cv2.dnn.readNetFromCaffe(
        '/content/drive/MyDrive/Cursos - recursos/Computer Vision Masterclass/Weights/pose_deploy_linevec_faster_4_stages.prototxt',
        '/content/drive/MyDrive/Cursos - recursos/Computer Vision Masterclass/Weights/pose_iter_160000.caffemodel')
    network.setInput(image_blob)
    output = network.forward()
    position_width = output.shape[3]
    position_height = output.shape[2]
    num_points = 15
    points = []
    threshold = 0.1
    for i in range(num_points):
        confidence_map = output[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(confidence_map)
        x = int((image_grey.shape[1] * point[0]) / position_width)
        y = int((image_grey.shape[0] * point[1]) / position_height)

        if confidence > threshold:
            cv2.circle(image_grey, (x, y), 3, (0, 255, 0), thickness=-1)
            cv2.putText(image_grey, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .2, (0, 0, 255))
            cv2.putText(image_grey, "{}-{}".format(point[0], point[1]), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 0, 255))
            points.append((x, y))
        else:
            points.append(None)

    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(image_grey, cv2.COLOR_BGR2RGB))
    # detections = face_detector.detectMultiScale(image_grey, minSize=(105,106))
    # eye_detections = face_detector.detectMultiScale(image_grey)
    #
    # for (x,y, w,h) in eye_detections:
    #     print('\n--------------', eye_detections)
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255), 2)


    cv2.imshow('Video',frame)