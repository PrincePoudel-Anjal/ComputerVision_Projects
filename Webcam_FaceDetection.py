import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

ret = True
while ret == True:
    ret,image = webcam.read()
    image = cv2.resize(image, (800, 600))

    # Detecting Faces

    mp_face_detection = mp.solutions.face_detection  # mp.solutions.face_detection are submodules
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                         model_selection=0) as facedetection:  # Renaming the class mp-face_detection_FaceDetection() as facedetection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = facedetection.process(image_rgb)  # .process is the function related to the class facedetection

        if output.detections is not None:
            for i in output.detections:
                location_data = i.location_data  # location_data isthe parameter of the object(detection)
                bbox = location_data.relative_bounding_box
                x1, y1, h, w = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * 800)
                y1 = int(y1 * 600)
                h = int(h * 800)
                w = int(w * 600)

                image = cv2.rectangle(image, (x1, y1), (x1 + h, y1 + w), (0, 255, 0), 6)

    cv2.imshow('Window1', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





