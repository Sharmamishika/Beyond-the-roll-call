# FACE-RECOGNITION-BASED-ATTENDANCE-MANAGEMENT-SYSTEM
PROJECT ON FACE RECOGINTION BASED ATTENDANCE MANAGEMENT SYSTEM WHICH IS CALLED "BEYOND THE ROLL CALL" FULL BASED ON PYTHON.

This project focuses on managing attendance through face recognition and can be used by any institution, school or office. It is a desktop application which does not require network.

DeepFace library is used for recognizing faces and FaceNet512 model is used. Model is trained on pre-captured images. This dataset is picked up from Kaggle. The dataset is augmented to create synthetic images to test the model. Since FaceNet512 is a highly accurate model, its accuracy came out to be 89%. Euclidean Distance is measured between detected faces and the embeddings, if the euclidean distance is less than a ceratin threshold then the face will be recognized with the name and roll no. If it more than the threshold decided then the face would be unknown.

Following are the libraries (dependencies) used:
1. cv2 (OpenCV): Used for video capture, face detection, image manipulation, and displaying the webcam feed.
2. pickle: Used to load and save serialized data (like embeddings, roll numbers, and names) to a file. This allows the system to persist face embeddings between sessions.
3. pandas: Used for handling attendance data in the form of a DataFrame. It allows easy manipulation of data, such as adding attendance records, viewing them, and exporting them to a CSV file.
4. datetime: Used to record the timestamp when a student’s attendance is marked. It helps format the current date and time when saving attendance.
5. deepface: A deep learning library for face recognition. It provides pre-trained models and methods to extract face embeddings and compare them to identify known faces.
6. numpy: Used for numerical operations like calculating the Euclidean distance between face embeddings to match faces. It handles arrays efficiently.
7. os: Used to interact with the operating system. It helps check the existence of files and directories and manage file paths.
8. tkinter: A Python library for creating graphical user interfaces (GUIs). It provides windows, buttons, labels, and other widgets to create interactive elements for the attendance system.
9. PIL (Pillow): Used for image processing. In this case, it is used to resize and display background images in the Tkinter GUI.
10. threading: Allows the webcam to run in a separate thread, enabling the GUI to remain responsive while the webcam is active.
11. tkinter.filedialog: Used to open file dialog boxes, allowing the user to select a folder to save the exported attendance CSV file.

Following are the functionalities of the system:
1. Start camera: This button will start the webcam and detected faces will be recognized by the DeepFace and FaceNet512 showing their roll no and name. If the faces are not detected it will show unknown.
2. Stop camera: After recognizing face, to stop the camera, we have to trigger the button stop camera to stop the webcam.
3. View Attendance: This button will show us the attendance that could be daily, weekly or monthly along with the name, roll no and the timestamp at which the face was recognized.
4. Export Attendance: This will export the attendance recorded in a csv file format to any selected folder.
5. Clear Attendance: This button will simply clear all the recorded attendance.
6. Register New Face: This will register a new face. The person have to enter the roll no and name and then the webcam will be started to record the face so that the model can be trained on that face and can be converted to an embedding which would be saved in the pickle file. The person have to press "c" key to capture the face. After that, the face will be recognized when starting the camera.


**OUTPUT-**

Training the model:
![Screenshot 2025-04-30 165457](https://github.com/user-attachments/assets/41aef1be-4588-4880-8b55-30b53e05df5a)
![Screenshot 2025-04-30 165536](https://github.com/user-attachments/assets/bde33851-f90d-4856-8c99-a4f625238e66)
![Screenshot 2025-04-30 165711](https://github.com/user-attachments/assets/a016eeb3-5e10-4a97-89da-698c3f564639)

Testing the model:
![Screenshot 2025-04-30 195140](https://github.com/user-attachments/assets/a3302fd4-fb56-4e16-8d6e-359afd23c8b5)
![Screenshot 2025-04-30 195207](https://github.com/user-attachments/assets/9bd0429f-30d0-4b47-825b-98c13af5d3e4)
![Screenshot 2025-04-30 195346](https://github.com/user-attachments/assets/d7ec2ebd-9834-425e-9e0a-6825b0f8299b)
![Screenshot 2025-04-30 195254](https://github.com/user-attachments/assets/4d6dee74-c680-4ec2-a47d-92547c1366cf)

Deploying the system in desktop using tkinter:
![image](https://github.com/user-attachments/assets/4adcf583-ad99-426d-9580-9fa8a2fd16eb)
![image](https://github.com/user-attachments/assets/db83186f-e008-4e3f-9c1f-137020e854bb)
![image](https://github.com/user-attachments/assets/d3d0b62d-d28b-4df3-a717-a4fc692ba16e)
![image](https://github.com/user-attachments/assets/504f6836-1469-45d6-9026-bb2b0ac04457)
![image](https://github.com/user-attachments/assets/a0da534c-6b0d-401c-91b7-ed41fdb21096)












