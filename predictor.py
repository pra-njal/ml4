# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# model_best = load_model('face_model.h5')

# class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# face1 = "face1.pbtxt"
# face2 = "face2.pb"
# age1 = "age1.prototxt"
# age2 = "age2.caffemodel"
# gen1 = "gen1.prototxt"
# gen2 = "gen2.caffemodel"

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# gender_labels = ['Male', 'Female']

# face_net = cv2.dnn.readNet(face2, face1)
# age_net = cv2.dnn.readNet(age2, age1)
# gender_net = cv2.dnn.readNet(gen2, gen1)


# def predict_emotion_age_gender_live():
#     cap = cv2.VideoCapture(0)

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.7:
#                 x1 = int(detections[0, 0, i, 3] * frame.shape[1])
#                 y1 = int(detections[0, 0, i, 4] * frame.shape[0])
#                 x2 = int(detections[0, 0, i, 5] * frame.shape[1])
#                 y2 = int(detections[0, 0, i, 6] * frame.shape[0])

#                 face = frame[max(0, y1 - 15):min(y2 + 15, frame.shape[0] - 1),
#                        max(0, x1 - 15):min(x2 + 15, frame.shape[1] - 1)]

#                 face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                 face_resized = cv2.resize(face_gray, (48, 48))
#                 face_array = image.img_to_array(face_resized)
#                 face_array = np.expand_dims(face_array, axis=0)
#                 emotion_predictions = model_best.predict(face_array)
#                 emotion_label = class_names[np.argmax(emotion_predictions)]

#                 blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#                 gender_net.setInput(blob)
#                 gender_preds = gender_net.forward()
#                 gender = gender_labels[gender_preds[0].argmax()]

#                 age_net.setInput(blob)
#                 age_preds = age_net.forward()
#                 age = age_labels[age_preds[0].argmax()]

#                 label = f'{gender}, {age}, {emotion_label}'
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.imshow('Emotion, Age, Gender Prediction', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     predict_emotion_age_gender_live()
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Predictor:
    def __init__(self):
        self.model_best = load_model('face_model.h5')
        self.class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        face1 = "face1.pbtxt"
        face2 = "face2.pb"
        age1 = "age1.prototxt"
        age2 = "age2.caffemodel"
        gen1 = "gen1.prototxt"
        gen2 = "gen2.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_labels = ['Male', 'Female']

        self.face_net = cv2.dnn.readNet(face2, face1)
        self.age_net = cv2.dnn.readNet(age2, age1)
        self.gender_net = cv2.dnn.readNet(gen2, gen1)

    def predict(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * frame.shape[0])

                face = frame[max(0, y1 - 15):min(y2 + 15, frame.shape[0] - 1),
                       max(0, x1 - 15):min(x2 + 15, frame.shape[1] - 1)]

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_array = image.img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0)
                emotion_predictions = self.model_best.predict(face_array)
                emotion_label = self.class_names[np.argmax(emotion_predictions)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender = self.gender_labels[gender_preds[0].argmax()]

                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = self.age_labels[age_preds[0].argmax()]

                results.append({
                    'position': (x1, y1, x2, y2),
                    'label': f'{gender}, {age}, {emotion_label}'
                })

        return results
