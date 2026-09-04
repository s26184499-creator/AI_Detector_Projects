# Import OpenCV for image processing
import cv2

# Import randrange to generate random values if needed for color (not used here but good to keep)
from random import randrange

# Load the pre-trained face detector (Haar Cascade)
trained_face_data = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Load the pre-trained smile detector
trained_smile_data = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

# Initialize webcam capture (0 is the default webcam)
webcam = cv2.VideoCapture(0)

# Loop to read frames continuously from webcam
while True:

    # Read a frame from the webcam
    successful_frame_read, frame = webcam.read()

    # Convert the frame to grayscale (needed for face/smile detection)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # detectMultiScale returns rectangles (x, y, w, h) for each detected face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Loop through each detected face
    for (x, y, w, h) in face_coordinates:
        # Draw a rectangle around the face (blue color, thickness 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region (ROI) from the color frame
        the_face = frame[y:y + h, x:x + w]

        # Convert the extracted face region to grayscale for smile detection
        face_img = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect smiles in the face region
        # scaleFactor: image size reduction, minNeighbors: required neighbor rectangles for detection
        smile = trained_smile_data.detectMultiScale(
            face_img,
            scaleFactor=1.7,
            minNeighbors=20
        )

        # Optionally draw rectangles around detected smiles (commented out)
        # for (x1, y1, w1, h1) in smile:
        #     cv2.rectangle(the_face, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)

        # Display smile detection result below the face
        if len(smile) > 0:
            # If smile(s) detected, show green text "smiling"
            cv2.putText(frame, 'smiling', (x, y + h + 40),
                        fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(0, 255, 0))
        else:
            # If no smile detected, show red text "not smiling"
            cv2.putText(frame, 'not smiling', (x, y + h + 40),
                        fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(0, 0, 255))

    # Print face coordinates (for debugging purposes)
    print(face_coordinates)

    # Show the final frame with drawings and text
    cv2.imshow('clever program face detector', frame)

    # Wait for 1 ms to detect key press
    key = cv2.waitKey(1)

    # If Q or q is pressed (ASCII 81 or 113), exit loop
    if key == 81 or key == 113:
        # Release webcam and close windows
        webcam.release()
        cv2.destroyAllWindows()
        break

# Final message after exiting loop
print("Done")
