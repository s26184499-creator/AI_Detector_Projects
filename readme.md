# AI Vision Project 👁️

This is a Python OpenCV-based project for real-time and video-based object detection. It includes face, smile, full-body, and car detection using Haarcascade classifiers.

---

## 📁 Folder Structure

```

AI/
├── datasets/
│   ├── images/                 # Sample images
│   └── videos/                 # Video files for detection
├── cascades/                   # Haar cascade XML files
├── src/                        # All Python detection scripts
├── projects/                   # Scratch/test projects
├── notes/                      # Documentation files
└── README.md                   # Project info

````

---

## ✅ Features

- Face detection (image, video, webcam)
- Smile detection (real-time webcam)
- Car and full-body detection
- Works with images and `.mp4` videos

---

## 🛠️ Requirements

- Python 3.x
- OpenCV library

Install OpenCV using pip:
```bash
pip install opencv-python
````

---

## 📂 Haar Cascade Files

Place these files in the `cascades/` folder:

* haarcascade\_frontalface\_default.xml
* haarcascade\_smile.xml
* haarcascade\_fullbody.xml
* car\_detector.xml

Download from:
[https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)

---

## ▶️ How to Use

### Live Face & Smile Detection (webcam):

```bash
python src/face_smile_detector.py
```

### Face Detection on Images:

```bash
python src/img_face_detector.py
```

### Face Detection on Video:

```bash
python src/video_face_detector.py
```

---

## 📌 Notes

* Press `Q` or `q` to exit any live webcam window.
* Modify `cv2.VideoCapture("videos/yourfile.mp4")` for custom videos.
* For best results, ensure good lighting when using a webcam.

---

## 🧠 Credits

Uses OpenCV and Haarcascade models from the OpenCV community.
Created as part of a computer vision learning project.

```

---

Creed Infotech
```
