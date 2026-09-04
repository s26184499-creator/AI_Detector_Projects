# import cv2

# from random import randrange

# trained_eye_data = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# webcame_access = cv2.VideoCapture(0)

# while True:
#     # Read the frames
#     successfully_read_frames , frame = webcame_access.read()
#     # Convert it to greyscale
#     greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Find face coordinates
#     face_coordinates =  trained_eye_data.detectMultiScale(greyscale_img)

#      # Draw rectangles around each detected face
#     for (x, y, w, h) in face_coordinates:
#             # Use a random color for the rectangle and thickness of 2
#         cv2.rectangle(frame, (x, y), (x + w, y + h), 
#         (randrange(256), randrange(256), randrange(256)), 2)

#     print(face_coordinates)


#     print(face_coordinates)

#     # Show the frame with rectangles in a window titled "clever program face detector"
#     cv2.imshow('clever program face detector', frame)

#     # Wait 1 millisecond for a key press and store the pressed key code
#     key = cv2.waitKey(1)

#     # If the key pressed is 'Q' or 'q' (ASCII codes 81 and 113), exit the loop
#     if key == 81 or key == 113:
#         # Release the webcam so it's available for other apps
#         webcame_access.release()

#         # Close all OpenCV windows
#         cv2.destroyAllWindows()

#         # Exit the loop
#         break
# print("Done")

















import cv2
from random import randrange

# ---------------------------------------------------------
# Load the Haar Cascade that comes with OpenCV
# ---------------------------------------------------------

cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

print("Loading Haar Cascade from:")
print(cascade_path)

trained_eye_data = cv2.CascadeClassifier(cascade_path)

# # Check if the cascade loaded correctly
# if trained_eye_data.empty():
#     print("ERROR: Haar Cascade could not be loaded.")
#     print("Path:", cascade_path)
#     exit()

# print("Haar Cascade loaded successfully!")


# ---------------------------------------------------------
# Open the webcam
# ---------------------------------------------------------

webcam_access = cv2.VideoCapture(0)

if not webcam_access.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully!")
print("Press Q to quit.")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:

    # Read a frame from the webcam
    successfully_read_frames, frame = webcam_access.read()

    # Check if frame was read successfully
    if not successfully_read_frames:
        print("ERROR: Could not read frame.")
        break

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


# ---------------------------------------------------------
# Clean up
# ---------------------------------------------------------

webcam_access.release()
cv2.destroyAllWindows()

print("Done")
