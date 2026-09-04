# Import the OpenCV library for image processing
import cv2

# Import 'randrange' function to generate random numbers for rectangle colors
from random import randrange

# Load the pre-trained face detection classifier (Haar Cascade)
# This XML file contains the data used to detect faces
trained_face_data = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Start video capture using the default webcam (device index 0)
webcam = cv2.VideoCapture(0)

# Run a loop to continuously read frames from the webcam
while True:    
    # Read a single frame from the webcam
    successful_frame_read, frame = webcam.read()

    # Convert the captured frame to grayscale (face detection works better on grayscale images)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the trained data
    # Returns a list of rectangles with (x, y, width, height) for each detected face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around each detected face
    for (x, y, w, h) in face_coordinates:
        # Use a random color for the rectangle and thickness of 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), 
                      (randrange(256), randrange(256), randrange(256)), 2)

    # Print coordinates of detected faces to the console (for debugging)
    print(face_coordinates)

    # Show the frame with rectangles in a window titled "clever program face detector"
    cv2.imshow('clever program face detector', frame)

    # Wait 1 millisecond for a key press and store the pressed key code
    key = cv2.waitKey(1)

    # If the key pressed is 'Q' or 'q' (ASCII codes 81 and 113), exit the loop
    if key == 81 or key == 113:
        # Release the webcam so it's available for other apps
        webcam.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Exit the loop
        break

# Print final confirmation that program has ended
print("Done")
