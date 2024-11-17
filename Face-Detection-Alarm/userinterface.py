import os
import cv2
import face_recognition
import numpy as np
import math
import datetime
import winsound
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import threading

# حساب الثقة
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2))
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2))

# قائمة فارغة للاسماء وموقع الوجه وميزات
# تحميل الاسماء والمواقع والوجوه


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
#تحميل من الدليل للقوائم
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encodings = face_recognition.face_encodings(face_image)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(os.path.splitext(image)[0])
            else:
                print(f"No face found in {image}. Skipping...")

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

    def delete_face(self, name):
        # Delete the face image and re-encode faces
        face_image_path = f'faces/{name}.jpg'
        if os.path.exists(face_image_path):
            os.remove(face_image_path)
            print(f"{name} has been deleted from the database.")
            self.encode_faces()  # Re-encode the faces after deleting the person
            messagebox.showinfo("Image Deleted", f"{name}'s image has been deleted successfully.")
        else:
            print(f"{name} not found in the database. No action taken.")

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
                messagebox.showinfo("Success")

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
        self.unknown_face_detected = False  # Flag to track unknown face detection
        self.recording_started = False  # Flag to track recording status
    #
    # def play_beep_sound(self):
    #     try:
    #         winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
    #     except:
    #         print("Error playing the beep sound.")

    def run_recognition(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            sys.exit('Video source not found ...')

        recording = False
        UNKNOWN_FACE_THRESHOLD = 50  # Set your desired threshold value here

        frame_size = (int(cap.get(3)), int(cap.get(4)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        while True:
            ret, frame = cap.read()
            if self.unknown_face_detected:
                winsound.Beep(2500, 50)

            if self.unknown_face_detected and len(recognized_names) > 0:
                self.unknown_face_detected = False  # Reset the flag if any known face is detected
                messagebox.showinfo("Unknown Face Stopped", "Unknown face no longer detected. Recording stopped.")

                if out is not None:
                    out.release()
                    recording = False  # Reset the recording state to False

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                recognized_names = []  # List to store recognized names

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        # If the face is recognized and the confidence level is higher than the set threshold
                        if float(confidence.split()[0]) >= UNKNOWN_FACE_THRESHOLD:
                            recognized_names.append(name)  # Add the recognized name to the list

                    self.face_names.append(f'{name} ({confidence})')

                # Start recording when an unknown face is detected,and it's not recognized
                if not recording and len(recognized_names) == 0 and len(self.face_names) > 0:
                    self.unknown_face_detected = True  # Set the flag for unknown face detection
                    print("Unknown face detected. Recording started...")
                    print("Unknown")

                    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
                    recording = True

                # Stop recording when a known face is detected or the unknown face disappears
                if recording and (recognized_names or len(self.face_names) == 0):
                    print("Recording stopped.")
                    recording = False
                    self.unknown_face_detected = False  # Reset the flag for unknown face detection
                    if out is not None:
                        out.release()

                # Check if the unknown face was registered during recording
                if self.unknown_face_detected and len(recognized_names) > 0:
                    self.unknown_face_detected = False  # Reset the flag if any known face is detected
                    print("Unknown face no longer detected. Recording stopped.")
                    if out is not None:
                        out.release()
                        recording = False  # Reset the recording state to False

                # Display annotations
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                # Write frame to the video if recording is in progress
                if recording:
                    out.write(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key == 27:  # 27 is the ASCII value of the ESC (Escape) key
                break
            elif key == ord('c'):
                self.save_captured_face(frame)

        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.fr = FaceRecognitionWithRecording()
        self.root.geometry("828x459")

        self.root.configure(bg='lightgray')  # تعيين خلفية النافذة

        # إنشاء صورة خلفية
        background_image = Image.open("E.jpg")  # استبدل ".jpg" بمسار صورة الخلفية المطلوبة
        background_image = background_image.resize((828, 459))  # تحديد حجم الصورة المناسب للنافذة
        self.background_photo = ImageTk.PhotoImage(background_image)

        # إنشاء عنصر Label لوضع الصورة كخلفية
        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # تحديد موقع العنصر ومساحته النسبية

        # self.video_label = tk.Label(root)
        # self.video_label.pack(pady=40)
        self.take_picture_button = tk.Button(root, text="Take Picture", command=self.take_picture, width=15, height=2, bg='black', fg='white')
        self.take_picture_button.pack(pady=40)
        self.take_picture_button.place(x=50, y=200)

        self.start_recognition_button = tk.Button(root, text="Start Recognition", command=self.start_recognition, width=15, height=2, bg='black', fg='white')
        self.start_recognition_button.place(x=50, y=300)

        self.delete_button = tk.Button(root, text="Delete", command=self.show_names_window, width=15, height=2,
                                       bg='black', fg='white')
        self.delete_button.place(x=50, y=400)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def take_picture(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key % 256 == 27:
                print('escape hit, closing the app')
                break

            elif key % 256 == 32:  # 32 is the ASCII value of the Spacebar key
                name = simpledialog.askstring("Input", "Enter the name of the person:")
                if name:
                    face_image_path = f'faces/{name}.jpg'
                    cv2.imwrite(face_image_path, frame)
                    print(f"Captured face for {name} saved successfully!")

                    # Show the success message box
                    messagebox.showinfo("Success", f"Image for {name} added successfully!")

                    self.fr.encode_faces()  # Re-encode the faces after adding the new person
                break

        cap.release()
        cv2.destroyAllWindows()

    def show_names_window(self):
        names_window = tk.Toplevel(self.root)
        names_window.title("Names in Database")
        names_window.geometry("400x300")
        names_window.configure(bg='lightgray')
        names_label = tk.Label(names_window, text="Names in Database:", font=("Arial", 14), bg='lightgray')
        names_label.pack(pady=10)
        names_listbox = tk.Listbox(names_window, font=("Arial", 12), selectbackground='gray')
        names_listbox.pack(fill='both', expand=True)
        # Fill the listbox with known names from the database
        for name in self.fr.known_face_names:
            names_listbox.insert(tk.END, name)

        def delete_selected_name():
            selected_index = names_listbox.curselection()
            if selected_index:
                selected_name = names_listbox.get(selected_index)
                if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {selected_name}?"):
                    self.fr.delete_face(selected_name)  # Call the new method to delete the selected face
                    names_listbox.delete(selected_index)
        delete_button = tk.Button(names_window, text="Delete Selected", command=delete_selected_name, width=15,
                                  height=2, bg='red', fg='white')
        delete_button.pack(pady=10)

    def start_recognition(self):

        self.face_recognition_thread = threading.Thread(target=self.fr.run_recognition)
        self.face_recognition_thread.start()

        # Show a message when an unknown face is recognized
        while self.fr.unknown_face_detected:
            messagebox.showwarning("Unknown Face Detected", "An unknown face has been detected!")
            self.fr.unknown_face_detected = False

    def on_closing(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("828x459")
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()



