# Import OpenCV for image processing
import cv2

# Import randrange if you want random colors (not used here currently)
from random import randrange

# Load Haarcascade classifiers for cars and full body detection
trained_face_data = cv2.CascadeClassifier('cascades/car_detector.xml')
trained_body_data = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

# Load video file (can be .mp4, .avi, etc.)
webcam2 = cv2.VideoCapture('datasets/videos/people02.mp4')

# Process video frame-by-frame
while True:
    # Read one frame from the video
    (read_successful, frame) = webcam2.read()

    # If frame read successfully, proceed
    if read_successful:
        # Convert the frame to grayscale for better detection performance
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # End of video or error reading frame
        break

    # Detect cars in the frame
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Detect full bodies in the frame
    face_coordinates2 = trained_body_data.detectMultiScale(grayscaled_img)

    # Draw red rectangles around detected cars
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw blue rectangles around detected full bodies
    for (x, y, w, h) in face_coordinates2:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Print coordinates for debugging
    print("Cars:", face_coordinates)
    print("Bodies:", face_coordinates2)

    # Display the annotated frame
    cv2.imshow('clever program face detector', frame)

    # Wait for 1 ms and check if user pressed 'Q' or 'q' to quit
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release the video and close OpenCV windows
webcam2.release()
cv2.destroyAllWindows()

print("Done")
