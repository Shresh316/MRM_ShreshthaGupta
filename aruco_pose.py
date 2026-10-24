
import os
import cv2
import numpy as np

# Marker and camera settings
MARKER_SIZE_CM = 5.0
CAM_INDEX = 0
CALIB_FILE = "calibration.npz"

# Marker dictionaries to check
tag_types = {
    "4x4": cv2.aruco.DICT_4X4_250,
    "5x5": cv2.aruco.DICT_5X5_250,
    "6x6": cv2.aruco.DICT_6X6_250
}

# Create one detector for each dictionary
detectors = {}

for name, tag_type in tag_types.items():
    dictionary = cv2.aruco.getPredefinedDictionary(tag_type)
    params = cv2.aruco.DetectorParameters()
    detectors[name] = cv2.aruco.ArucoDetector(dictionary, params)


def write_text(img, text, x, y, color=(0, 255, 0), size=0.5):
    # Add an outline so the text is easy to read
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, size,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, size,
        color, 1, cv2.LINE_AA
    )


def get_camera_settings(width, height):
    # Load camera calibration if available
    if os.path.exists(CALIB_FILE):
        try:
            with np.load(CALIB_FILE) as data:
                camera_matrix = data["K"].astype(np.float64)
                distortion = data["dist"].astype(np.float64)

            if camera_matrix.shape == (3, 3):
                print("Camera calibration loaded.")
                return camera_matrix, distortion

            print("Calibration matrix is invalid.")

        except (OSError, KeyError, ValueError) as error:
            print("Could not load calibration:", error)

    # Use an approximate camera matrix if calibration is missing
    print("Using approximate camera settings.")
    print("Position and distance may be inaccurate.")

    camera_matrix = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((5, 1), dtype=np.float64)

    return camera_matrix, distortion


def get_marker_corners(size_cm):
    # Set the four 3D corners of the marker
    half = size_cm / 2

    points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    return points


def get_pose(marker_2d, marker_3d, camera_matrix, distortion):
    # Calculate the marker's position and rotation
    ok, rvec, tvec = cv2.solvePnP(
        marker_3d,
        marker_2d,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not ok:
        return None

    # Convert rotation into angles
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    x = float(tvec[0, 0])
    y = float(tvec[1, 0])
    z = float(tvec[2, 0])

    # Calculate distance from the camera
    distance = np.sqrt(x**2 + y**2 + z**2)

    return rvec, tvec, x, y, z, distance, angles


def main():
    # Open the webcam
    cap = cv2.VideoCapture(CAM_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    try:
        # Read one frame to get the image size
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            return

        height, width = frame.shape[:2]

        camera_matrix, distortion = get_camera_settings(
            width, height
        )

        marker_3d = get_marker_corners(MARKER_SIZE_CM)

        while True:
            # Capture the next frame
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Keep marker information for the side panel
            marker_info = []

            # Check each marker dictionary
            for name, detector in detectors.items():
                corners, ids, rejected = detector.detectMarkers(gray)

                if ids is None:
                    continue

                # Draw the detected marker outlines
                cv2.aruco.drawDetectedMarkers(
                    frame, corners, ids
                )

                for i, marker_id in enumerate(ids.flatten()):
                    marker_2d = corners[i][0].astype(np.float32)

                    # Estimate the marker pose
                    pose = get_pose(
                        marker_2d,
                        marker_3d,
                        camera_matrix,
                        distortion
                    )

                    if pose is None:
                        continue

                    rvec, tvec, x, y, z, distance, angles = pose

                    # Draw the marker's X, Y and Z axes
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        distortion,
                        rvec,
                        tvec,
                        MARKER_SIZE_CM * 0.5
                    )

                    # Put the label near the marker
                    tx = max(5, int(marker_2d[:, 0].min()))
                    ty = max(20, int(marker_2d[:, 1].min()) - 10)

                    write_text(
                        frame, f"{name} ID: {marker_id}",
                        tx, ty, (0, 255, 0)
                    )

                    marker_info.append({
                        "name": name,
                        "id": int(marker_id),
                        "x": x,
                        "y": y,
                        "z": z,
                        "distance": distance,
                        "angles": angles
                    })

            # Show the marker axes guide
            write_text(
                frame, "X: Red  Y: Green  Z: Blue",
                10, 25, (255, 255, 255), 0.5
            )
            write_text(
                frame, "Press Q to quit",
                10, 48, (255, 255, 255), 0.45
            )

            # Make the information panel
            panel = np.full(
                (height, 340, 3), 30, dtype=np.uint8
            )

            write_text(
                panel, "ARUCO POSE ESTIMATION",
                12, 28, (255, 255, 255), 0.55
            )

            cv2.line(
                panel, (12, 40), (328, 40),
                (90, 90, 90), 1
            )

            if not marker_info:
                write_text(
                    panel, "No markers detected",
                    12, 70, (150, 150, 150)
                )
                write_text(
                    panel, "Show a supported marker",
                    12, 95, (200, 200, 200), 0.42
                )

            # Display information for each marker
            y_pos = 65

            for info in marker_info:
                if y_pos > height - 125:
                    write_text(
                        panel, "More markers not shown",
                        12, y_pos, (150, 150, 150), 0.4
                    )
                    break

                name = info["name"]
                marker_id = info["id"]
                angles = info["angles"]

                write_text(
                    panel, f"{name} marker - ID {marker_id}",
                    12, y_pos, (0, 255, 120), 0.48
                )
                y_pos += 23

                write_text(
                    panel, f"X: {info['x']:.1f} cm",
                    12, y_pos, (100, 220, 255), 0.43
                )
                y_pos += 19

                write_text(
                    panel, f"Y: {info['y']:.1f} cm",
                    12, y_pos, (100, 220, 255), 0.43
                )
                y_pos += 19

                write_text(
                    panel, f"Depth Z: {info['z']:.1f} cm",
                    12, y_pos, (100, 220, 255), 0.43
                )
                y_pos += 19

                write_text(
                    panel, f"Distance: {info['distance']:.1f} cm",
                    12, y_pos, (0, 200, 255), 0.43
                )
                y_pos += 19

                write_text(
                    panel,
                    f"Rotation: {angles[0]:.0f}, "
                    f"{angles[1]:.0f}, {angles[2]:.0f} deg",
                    12, y_pos, (255, 220, 100), 0.40
                )
                y_pos += 28

            # Show the camera and information panel
            display = np.hstack((frame, panel))
            cv2.imshow("Task 1 - ArUco Pose", display)

            # Quit when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
