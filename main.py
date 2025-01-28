import sys
import ctypes
import os
from tkinter import *
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
import easyocr
from ultralytics import YOLO
from sort import Sort
from util import get_car, estimate_speed
from time import sleep

# Set environment variable to avoid potential warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'en'])

# Initialize SORT tracker
mot_tracker = Sort()
prev_bbox_centers = {}

# Load YOLO models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO('license_plate_detector.pt')

# Global variable to store detection results
results = {}


def browsevideo():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                          filetypes=(("Video Files", (".mp4", ".avi", ".mkv")), ("All Files", "*.*")))

    # Load video
    cap = cv2.VideoCapture(filename)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    vehicles = [2, 3, 5, 7]

    frame_nmr = -1

    # Define font and color for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 255, 0)  # Green color for text

    # Set up overspeed folder
    overspeed_folder = "overspeed"
    if not os.path.exists(overspeed_folder):
        os.makedirs(overspeed_folder)

    # Read frames
    prev_frame = None
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

                    # Draw rectangle around detected vehicle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Draw rectangle around detected license plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    img_text = reader.readtext(license_plate_crop)
                    final_text = ""
                    for _, text, __ in img_text:  # _ = bounding box, text = text and __ = confident level
                        final_text += " "
                        final_text += text

                    car_data = {
                        'locations': track_ids,  # Assuming you have the car's locations over time in track_ids
                    }
                    car_info = estimate_speed(car_id, car_data)
                    # Display speed and license plate number on the frame
                    cv2.putText(frame, f"Speed: {car_info['speed_label']}", (int(x1), int(y1) - 20), font,
                                font_scale, text_color, font_thickness)
                    cv2.putText(frame, f"License Plate: {final_text}", (int(x1), int(y1) + 20), font,
                                font_scale, text_color, font_thickness)

                    # Check for overspeed and save cropped car image
                    speed_label = car_info['speed_label']
                    speed_value = speed_label.split('km/h')[0].strip()  # Extract numerical part
                    if speed_value.isdigit() and int(speed_value) > 30:
                        # Crop the car from the frame
                        car_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]
                        # Save the cropped car image in the overspeed folder
                        filename = f"{overspeed_folder}/car_{frame_nmr}_speed_{speed_value}.jpg"
                        cv2.imwrite(filename, car_crop)

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Release video capture
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def exit():
    result = tkMessageBox.askquestion("Smart Street Pole", "Are you sure you want to exit?", icon="warning")
    if result == 'yes':
        sys.exit()


def HomePage():
    root = Tk()
    img = Image.open("Images\\HomePage.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0] // 2 - 446)
    b = str(lt[1] // 2 - 383)

    root.title("HOME - Smart Street Pole")
    root.geometry("904x533+" + a + "+" + b)
    root.resizable(0, 0)

    def aboutus():
        about = Tk()
        img = Image.open("Images\\AboutUs.png")
        img = ImageTk.PhotoImage(img)
        panel = Label(about, image=img)
        panel.pack(side="top", fill="both", expand="yes")

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        lt = [w, h]
        a = str(lt[0] // 2 - 446)
        b = str(lt[1] // 2 - 383)

        about.title("ABOUT US - Smart Street Pole")
        about.geometry("904x533+" + a + "+" + b)
        about.resizable(0, 0)

        homebtn = Button(about, text="HomePage", font=("Agency FB", 16, "bold"), relief=FLAT, bd=0, borderwidth='0',
                         bg="#1A1D24", fg="#D5DAE0", activebackground="#1A1D24", activeforeground="#D5DAE0",
                         command=root.destroy)
        homebtn.place(x=674, y=20)

        exitbtn = Button(about, text="Exit", font=("Agency FB", 16, "bold"), relief=FLAT, bd=0, borderwidth='0',
                         bg="#191C23", fg="#D5DAE0", activebackground="#191C23", activeforeground="#D5DAE0",
                         command=exit)
        exitbtn.place(x=782, y=20)

    aboutusbtn = Button(root, text="About Us", font=("Agency FB", 16, "bold"), relief=FLAT, bd=0, borderwidth='0',
                        bg="#000000", fg="#948B8B", activebackground="#000000", activeforeground="#948B8B",
                        command=aboutus)
    aboutusbtn.place(x=674, y=10)

    exitbtn = Button(root, text="Exit", font=("Agency FB", 16, "bold"), relief=FLAT, bd=0, borderwidth='0',
                     bg="#000000", fg="#948B8B", activebackground="#000000", activeforeground="#948B8B",
                     command=exit)
    exitbtn.place(x=782, y=10)

    videobtn = Button(root, text="UPLOAD VIDEO", font=("Arial Narrow", 16, "bold"), width=15, relief=FLAT, bd=1,
                      borderwidth='1', bg="#59070F", fg="#807676", activebackground="#59070F",
                      activeforeground="#807676", command=browsevideo)
    videobtn.place(x=75, y=342)

    root.mainloop()


def LoadingScreen():
    root = Tk()
    root.config(bg="white")
    root.title("Loading - Smart Street Pole")

    img = Image.open(r"Images\\Loading.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0] // 2 - 446)
    b = str(lt[1] // 2 - 383)

    root.geometry("902x553+" + a + "+" + b)
    root.resizable(0, 0)

    for i in range(27):
        Label(root, bg="#574D72", width=2, height=1).place(x=(i + 4) * 25, y=520)

    def play_animation():
        for j in range(27):
            Label(root, bg="#9477CD", width=2, height=1).place(x=(j + 4) * 25, y=520)
            sleep(0.17)
            root.update_idletasks()
        else:
            root.destroy()
            HomePage()

    root.update()
    play_animation()
    root.mainloop()


LoadingScreen()
