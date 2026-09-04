import cv2
from random import randrange

# Load the Haar Cascade that comes with OpenCV

cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

print("Loading Haar Cascade from:")
print(cascade_path)

trained_eye_data = cv2.CascadeClassifier(cascade_path)

# ---------------------------------------------------------
# Open the webcam
# ---------------------------------------------------------

webcam_access = cv2.VideoCapture(0)


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:

    # Read a frame from the webcam
    successfully_read_frames, frame = webcam_access.read()


    # Convert frame to grayscale
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eye_coordinates = trained_eye_data.detectMultiScale(
        greyscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Draw rectangles around detected eyes
    for (x, y, w, h) in eye_coordinates:

        # Random rectangle color
        color = (
            randrange(256),
            randrange(256),
            randrange(256)
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    # Display number of detected eyes
    cv2.putText(
        frame,
        f"Eyes detected: {len(eye_coordinates)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the webcam
    cv2.imshow(
        "Clever Program Eye Detector",
        frame
    )

    # Wait for keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Press Q to quit
    if key == ord("q"):
        break



# Clean up

webcam_access.release()
cv2.destroyAllWindows()

print("Done")
