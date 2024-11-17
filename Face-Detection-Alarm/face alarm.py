import face_recognition
import os
import cv2
import numpy as np
import math
import time
import datetime
import winsound
import sys


def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2))
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2))


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(os.path.splitext(image)[0])

        print(self.known_face_names)

    def save_captured_face(self, frame):
        name = input("Enter the name of the person: ")
        face_image_path = f'faces/{name}.jpg'
        cv2.imwrite(face_image_path, frame)
        print(f"Captured face for {name} saved successfully!")

        # Re-encode the faces after adding the new person
        face_image = face_recognition.load_image_file(face_image_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]

        # Check if the captured face is recognized before adding it to the known faces
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        if any(matches):
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            print(f"Face recognized as {name}.")
        else:
            print("Face not recognized. The captured face will not be added.")

    def run_recognition(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # find all faces in the current frame
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            confidence = 'Unknown'

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append(f'{name} ({confidence})')

        # Display annotations
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame


class FaceRecognitionWithRecording(FaceRecognition):
    def __init__(self):
        super().__init__()

    def run_recognition(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            sys.exit('Video source not found ...')

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml")

        detection = False
        detection_time = None
        timer_started = False

        SECONDS_TO_WAIT_BEFORE_RECORDING = 5

        SECONDS_TO_RECORD_AFTER_DETECTION = 5

        UNKNOWN_FACE_THRESHOLD = 50  # Set your desired threshold value here

        frame_size = (int(cap.get(3)), int(cap.get(4)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        unknown_face_detected = False  # Initialize the variable outside the loop

        while True:
            ret, frame = cap.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                recognized_names = []  # قائمة لتخزين الأسماء المعروفة المعترف بها

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        # إذا تم التعرف على الوجه وكان مستوى الثقة أعلى من مستوى الثقة المحدد
                        if float(confidence.split()[0]) >= UNKNOWN_FACE_THRESHOLD:
                            recognized_names.append(name)  # أضف اسم الوجه المعروف للقائمة

                    self.face_names.append(f'{name} ({confidence})')


                    # Check if any of the faces are unknown (confidence level below threshold)
                    unknown_face_detected = any(
                        confidence == 'Unknown' or float(confidence.split()[0]) < UNKNOWN_FACE_THRESHOLD for confidence
                        in self.face_names if confidence.split()[0].isdigit())

                # Start the timer when an unknown face is detected
                if unknown_face_detected and not timer_started and len(self.known_face_names) > 0:
                    timer_started = True
                    detection_time = time.time()
                    print("Unknown face detected. Waiting for a few seconds...")

                # Check if the timer has elapsed and start recording
                if timer_started and time.time() - detection_time >= SECONDS_TO_WAIT_BEFORE_RECORDING:
                    if not detection:
                        detection = True
                        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        out = cv2.VideoWriter(
                            f"{current_time}.mp4", fourcc, 20, frame_size)
                        print("Starting recording...")
                        # winsound.Beep(2500, 1000)

                # Check if any face is recognized to stop the recording
                if detection and not unknown_face_detected:
                    if timer_started and time.time() - detection_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                        detection = False
                        timer_started = False
                        out.release()
                        print("Stopped recording.")
                        # winsound.Beep(2500, 1000)

                # Display annotations
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                    # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    # Check if any face is recognized to display the appropriate message
                    if recognized_names:
                        print(f"Recognized faces: {', '.join(recognized_names)}")
                    else:
                        print("Not Recognized")
                        winsound.Beep(2500, 1000)  # Play alert sound when face is not recognized

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) + len(bodies) > 0:
                    if detection:
                        # winsound.Beep(2500, 1000)
                        timer_started = False
                    else:
                        detection = True
                        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        out = cv2.VideoWriter(
                            f"{current_time}.mp4", fourcc, 20, frame_size)
                        print("Started Recording!")
                        # winsound.Beep(2500, 1000)

                elif detection:
                    if timer_started:
                        if time.time() - detection_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                            detection = False
                            timer_started = False
                            out.release()
                            print('Stop Recording!')
                    else:
                        timer_started = True
                        detection_time = time.time()

                if detection:
                    out.write(frame)

                cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.save_captured_face(frame)

        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()


def main():
    fr = FaceRecognitionWithRecording()
    fr.run_recognition()


if __name__ == '__main__':
    main()



