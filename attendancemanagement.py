import cv2
import pickle
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from PIL import Image, ImageTk
from tkinter import filedialog


# Load trained embeddings
def load_embeddings():
    if os.path.exists("trained_embeddings.pkl"):
        with open("trained_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return {"embeddings": [], "roll_numbers": [], "names": []}

data = load_embeddings()
embeddings = data["embeddings"]
roll_numbers = data["roll_numbers"]
names = data["names"]

# Attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    attendance_df = pd.DataFrame(columns=["Roll Number", "Name", "Time"])
else:
    attendance_df = pd.read_csv(attendance_file)


def save_embeddings():
    with open("trained_embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "roll_numbers": roll_numbers, "names": names}, f)


def mark_attendance(roll_no, name):
    global attendance_df
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    
    if not ((attendance_df["Roll Number"] == roll_no) & (attendance_df["Name"] == name)).any():
        new_entry = pd.DataFrame([{"Roll Number": roll_no, "Name": name, "Time": time_string}])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv(attendance_file, index=False)
        print(f"✅ Attendance marked for {name} ({roll_no}) at {time_string}")
    else:
        print(f"🕒 {name} ({roll_no}) already marked present.")


def find_match(face_embedding):
    min_distance = float('inf')
    idx = -1
    for i, emb in enumerate(embeddings):
        emb1 = np.array(emb) / np.linalg.norm(emb)
        emb2 = np.array(face_embedding) / np.linalg.norm(face_embedding)
        dist = np.linalg.norm(emb1 - emb2)
        if dist < min_distance:
            min_distance = dist
            idx = i
    if min_distance < 0.7:
        return idx
    else:
        return -1
stop_flag = False


def start_webcam(window, label):
    global stop_flag
    stop_flag = False  
    cap = cv2.VideoCapture(0)
    
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            results = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            for face in results:
                face_img = face["face"]
                embedding_obj = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="skip"
                )
                face_embedding = embedding_obj[0]["embedding"]

                idx = find_match(face_embedding)

                if idx != -1:
                    roll_no = roll_numbers[idx]
                    name = names[idx]
                    mark_attendance(roll_no, name)

                    cv2.putText(frame, f"{roll_no} {name}",
                                (face["facial_area"]["x"], face["facial_area"]["y"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame,
                                  (face["facial_area"]["x"], face["facial_area"]["y"]),
                                  (face["facial_area"]["x"] + face["facial_area"]["w"], face["facial_area"]["y"] + face["facial_area"]["h"]),
                                  (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown - Press R",
                                (face["facial_area"]["x"], face["facial_area"]["y"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame,
                                  (face["facial_area"]["x"], face["facial_area"]["y"]),
                                  (face["facial_area"]["x"] + face["facial_area"]["w"], face["facial_area"]["y"] + face["facial_area"]["h"]),
                                  (0, 0, 255), 2)
        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("BEYOND THE ROLL CALL", frame)
        window.after(0, lambda: label.config(text="Running... Press 'q' to quit"))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def stop_webcam():
    global stop_flag
    stop_flag = True


def register_new_face_ui():
    reg_window = tk.Toplevel()
    reg_window.title("Register New Face")
    reg_window.geometry("300x200")

    tk.Label(reg_window, text="Roll Number:").pack()
    roll_entry = tk.Entry(reg_window)
    roll_entry.pack()
    tk.Label(reg_window, text="Name:").pack()
    name_entry = tk.Entry(reg_window)
    name_entry.pack()

    def capture_face():
        roll_no = roll_entry.get().strip()
        name = name_entry.get().strip()
        if not roll_no or not name:
            messagebox.showerror("Input Error", "Both fields are required.")
            return
        cap = cv2.VideoCapture(0)
        registered = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register Face - Press 'c' to capture", frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                try:
                    faces = DeepFace.extract_faces(frame, enforce_detection=True)
                    if faces:
                        face_img = faces[0]['face']
                        face_image_name = f"{roll_no}_{name}.png"
                        cv2.imwrite(face_image_name, face_img)
                        embedding_obj = DeepFace.represent(
                            img_path=face_img,
                            model_name="Facenet512",
                            enforce_detection=False,
                            detector_backend="skip"
                        )
                        face_embedding = embedding_obj[0]["embedding"]

                        embeddings.append(face_embedding)
                        roll_numbers.append(roll_no)
                        names.append(name)
                        save_embeddings()

                        messagebox.showinfo("Success", f"Registered {name} ({roll_no}) successfully!")
                        registered = True
                        break
                except Exception as e:
                    messagebox.showerror("Error", f"Face not detected or error: {e}")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if registered:
            reg_window.destroy()

    tk.Button(reg_window, text="Capture Face", command=capture_face).pack(pady=10)

# View Attendance
def view_attendance(window):
    new_window = tk.Toplevel(window)
    new_window.title("Attendance Records")
    new_window.geometry("600x400")
    new_window.configure(bg="white")

    tk.Label(new_window, text="🗂️ Attendance Records", font=("Helvetica", 16, "bold"), bg="white", fg="#333").pack(pady=10)
    attendance_df = pd.read_csv(attendance_file)
    frame = tk.Frame(new_window)
    frame.pack(fill="both", expand=True, padx=20, pady=10)
    canvas = tk.Canvas(frame, bg="white")
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="white")
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for i, row in attendance_df.iterrows():
        text = f"{row['Roll Number']} | {row['Name']} | {row['Time']}"
        tk.Label(scroll_frame, text=text, anchor="w",
                 font=("Courier New", 12), bg="white", fg="#222").pack(fill="x", pady=2)

    tk.Button(new_window, text="Close", command=new_window.destroy,
              font=("Helvetica", 11), bg="#f44336", fg="white",
              activebackground="#c0392b", relief="flat", padx=10, pady=5).pack(pady=10)

def export_attendance_csv():
    if attendance_df.empty:
        messagebox.showinfo("No Data", "No attendance data to export.")
        return

    folder_path = filedialog.askdirectory(title="Select Folder to Save CSV")
    if folder_path:
        export_path = os.path.join(folder_path, "exported_attendance.csv")
        attendance_df.to_csv(export_path, index=False)
        messagebox.showinfo("Success", f"Attendance exported to:\n{export_path}")


def clear_attendance():
    global attendance_df
    attendance_df = pd.DataFrame(columns=["Roll Number", "Name", "Time"])
    attendance_df.to_csv(attendance_file, index=False)
    messagebox.showinfo("Success", "Attendance cleared successfully!")


def create_ui():
    window = tk.Tk()
    window.title("Face Recognition Attendance System")
    window.geometry("500x400")
    window.configure(bg="#f2f2f2")
    bg_image_original = Image.open(r"C:\Users\Silky\OneDrive\Desktop\MINOR 2\Untitled design (1).png")
    bg_resized = bg_image_original.resize((500, 400), Image.ANTIALIAS)
    bg_photo = ImageTk.PhotoImage(bg_resized)
    background_label = tk.Label(window, image=bg_photo)
    background_label.image = bg_photo  
    background_label.place(relwidth=1, relheight=1)

    def update_bg(event):
        resized_bg = bg_image_original.resize((event.width, event.height), Image.ANTIALIAS)
        bg_photo_resized = ImageTk.PhotoImage(resized_bg)
        background_label.config(image=bg_photo_resized)
        background_label.image = bg_photo_resized

    window.bind("<Configure>", update_bg)

    heading = tk.Label(window, text="📸 BEYOND THE ROLL CALL", font=("Helvetica", 20, "bold"), bg="#f2f2f2", fg="#333")
    heading.pack(pady=20)

    label = tk.Label(window, text="Click a button to begin", font=("Helvetica", 14), bg="#f2f2f2")
    label.pack(pady=10)

    def styled_button(text, command):
        return tk.Button(window, text=text, command=command,
                         font=("Helvetica", 12),
                         bg="#1e3a5f", fg="white",  
                         activebackground="#1c3c57",
                         relief="raised", bd=3, padx=10, pady=5)
    
    button_frame = tk.Frame(window, bg="#f2f2f2")
    button_frame.pack(pady=20, fill="x", expand=True)

    left_buttons = tk.Frame(button_frame, bg="#f2f2f2")
    left_buttons.pack(side="left", padx=50, anchor="nw")

    styled_button("Register New Face", register_new_face_ui).pack(pady=10, anchor="w")
    styled_button("View Attendance", lambda: view_attendance(window)).pack(pady=10, anchor="w")
    styled_button("Clear Attendance", clear_attendance).pack(pady=10, anchor="w")
    styled_button("Export Attendance", export_attendance_csv).pack(pady=10, anchor="w")

    right_buttons = tk.Frame(button_frame, bg="#f2f2f2")
    right_buttons.pack(side="right", padx=50, anchor="ne")

    styled_button("Start Webcam", lambda: Thread(target=start_webcam, args=(window, label)).start()).pack(pady=10, anchor="e")
    styled_button("Stop Webcam", stop_webcam).pack(pady=10, anchor="e")

    window.mainloop()

if __name__ == "__main__":
    create_ui()
