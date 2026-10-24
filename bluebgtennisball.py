
import cv2
import numpy as np

# Camera and object sizes
CAMERA_ID = 0
MARKER_SIZE_CM = 5.0
BALL_SIZE_CM = 6.6

# Colour ranges (HSV)
BLUE_LOW = np.array([100, 100, 45])
BLUE_HIGH = np.array([135, 255, 255])

BALL_LOW = np.array([22, 70, 70])
BALL_HIGH = np.array([45, 255, 255])

MIN_BLUE_AREA = 500
KERNEL = np.ones((5, 5), np.uint8)

# ArUco marker setup
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict, aruco_params
)

# Values used while running
pixels_per_cm = None
scale_source = "Not found"
selected_point = None
selected_center = None
trail = []
travelled = 0.0
show_masks = False


def clean_mask(mask):
    # Remove noise and fill small gaps
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, KERNEL
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, KERNEL
    )
    return mask


def put_text(img, text, x, y, color=(255, 255, 255), size=0.5):
    # Draw readable text
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size, (20, 20, 20), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size, color, 1, cv2.LINE_AA
    )


def get_contours(mask):
    # Find object outlines
    result = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return result[0] if len(result) == 2 else result[1]


def length_text(distance):
    # Convert pixels to centimetres
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return f"{distance:.0f} px"
    return f"{distance / pixels_per_cm:.1f} cm"


def area_text(area):
    # Convert pixel area to square centimetres
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return f"{area:.0f} px^2"
    return f"{area / (pixels_per_cm ** 2):.1f} cm^2"


def find_blue_objects(mask):
    # Find blue areas large enough to be objects
    objects = []

    for contour in get_contours(mask):
        area = cv2.contourArea(contour)

        if area < MIN_BLUE_AREA:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(contour)

        objects.append({
            "contour": contour,
            "area": area,
            "center": (cx, cy),
            "box": (x, y, w, h)
        })

    # Number objects from left to right
    objects.sort(key=lambda item: item["center"][0])

    return objects


def find_ball(hsv):
    # Find round yellow-green areas
    mask = clean_mask(
        cv2.inRange(hsv, BALL_LOW, BALL_HIGH)
    )

    ball = None

    for contour in get_contours(mask):
        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius < 8:
            continue

        area = cv2.contourArea(contour)
        circle_area = np.pi * radius * radius

        if circle_area == 0:
            continue

        # Check how round the object is
        roundness = area / circle_area

        if roundness < 0.55:
            continue

        if ball is None or area > ball["area"]:
            ball = {
                "center": (int(x), int(y)),
                "radius": radius,
                "area": area
            }

    return mask, ball


def find_marker(gray):
    # Find ArUco markers and measure their sides
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None:
        return None, corners, ids

    side_lengths = []

    for marker in corners:
        points = marker[0]

        for i in range(4):
            side = np.linalg.norm(
                points[i] - points[(i + 1) % 4]
            )
            side_lengths.append(side)

    if not side_lengths:
        return None, corners, ids

    scale = np.mean(side_lengths) / MARKER_SIZE_CM

    return scale, corners, ids


def mouse_click(event, x, y, flags, param):
    # Left-click to select and right-click to clear
    global selected_point, selected_center, trail, travelled

    if event == cv2.EVENT_LBUTTONDOWN and x < 640:
        selected_point = (x, y)
        selected_center = None
        trail = []
        travelled = 0.0

    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_point = None
        selected_center = None
        trail = []
        travelled = 0.0


def get_position(center, ball_center):
    # Find the object's position relative to the ball
    dx = center[0] - ball_center[0]
    dy = center[1] - ball_center[1]

    if dx < -15:
        horizontal = "left"
    elif dx > 15:
        horizontal = "right"
    else:
        horizontal = "level"

    if dy < -15:
        vertical = "above"
    elif dy > 15:
        vertical = "below"
    else:
        vertical = "level"

    distance = np.hypot(dx, dy)

    return horizontal, vertical, distance, dx, dy


def main():
    global pixels_per_cm, scale_source
    global selected_point, selected_center
    global trail, travelled, show_masks

    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    window = "Task 3 - Object Tracker"

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_click)

    try:
        while True:
            # Read camera frame
            ok, frame = cap.read()

            if not ok:
                print("Could not read camera frame.")
                break

            # Prepare image
            blurred = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find tennis ball
            ball_mask, ball = find_ball(hsv)

            # Get scale from marker first
            marker_scale, corners, ids = find_marker(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    frame, corners, ids
                )

            if marker_scale is not None:
                new_scale = marker_scale
                scale_source = "ArUco"
            elif ball is not None:
                new_scale = (
                    2 * ball["radius"] / BALL_SIZE_CM
                )
                scale_source = "Tennis ball"
            else:
                new_scale = None
                scale_source = "Not found"

            # Smooth the scale
            if new_scale is not None and new_scale > 0:
                if pixels_per_cm is None:
                    pixels_per_cm = new_scale
                else:
                    pixels_per_cm = (
                        0.8 * pixels_per_cm
                        + 0.2 * new_scale
                    )

            # Find blue objects
            blue_mask = clean_mask(
                cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
            )
            objects = find_blue_objects(blue_mask)

            # Find the selected object
            chosen = None

            if selected_point is not None:
                for obj in objects:
                    inside = cv2.pointPolygonTest(
                        obj["contour"], selected_point, False
                    )

                    if inside >= 0:
                        chosen = obj
                        break

                # Keep following the object if it moves
                if chosen is None and selected_center is not None:
                    nearest = 120

                    for obj in objects:
                        cx, cy = obj["center"]

                        distance = np.hypot(
                            cx - selected_center[0],
                            cy - selected_center[1]
                        )

                        if distance < nearest:
                            nearest = distance
                            chosen = obj

            # Draw the tennis ball
            if ball is not None:
                bx, by = ball["center"]
                radius = int(ball["radius"])

                cv2.circle(
                    frame, (bx, by), radius,
                    (60, 220, 60), 2
                )
                cv2.circle(
                    frame, (bx, by), 4,
                    (60, 220, 60), -1
                )

                put_text(
                    frame, "TENNIS BALL",
                    max(5, bx - 45),
                    max(20, by - radius - 10),
                    (80, 255, 100)
                )

            # Draw all blue objects
            centers = []
            details = []

            for number, obj in enumerate(objects, start=1):
                contour = obj["contour"]
                cx, cy = obj["center"]
                x, y, w, h = obj["box"]
                area = obj["area"]

                is_selected = obj is chosen

                if is_selected:
                    color = (50, 80, 255)
                else:
                    color = (255, 150, 40)

                cv2.drawContours(
                    frame, [contour], -1, color, 2
                )
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 220, 255), 1
                )
                cv2.circle(
                    frame, (cx, cy), 4, color, -1
                )

                put_text(
                    frame, f"BLUE {number}",
                    x, max(18, y - 8), color
                )
                put_text(
                    frame, area_text(area),
                    x, min(465, y + h + 18), color, 0.45
                )

                centers.append((cx, cy))

                details.append(
                    (f"BLUE {number}: centre ({cx}, {cy})", color)
                )
                details.append(
                    (f"Area: {area_text(area)}", color)
                )

                # Position relative to tennis ball
                if ball is not None:
                    horizontal, vertical, distance, dx, dy = (
                        get_position((cx, cy), ball["center"])
                    )

                    position = f"{vertical}, {horizontal}"

                    details.append(
                        (f"From ball: {position}", color)
                    )
                    details.append(
                        (
                            f"X: {length_text(abs(dx))}, "
                            f"Y: {length_text(abs(dy))}",
                            color
                        )
                    )
                    details.append(
                        (f"Distance: {length_text(distance)}", color)
                    )

                    cv2.line(
                        frame, (cx, cy), ball["center"],
                        (220, 60, 220), 1
                    )

                # Track the selected object's movement
                if is_selected:
                    if selected_center is not None:
                        step = np.hypot(
                            cx - selected_center[0],
                            cy - selected_center[1]
                        )

                        if 3 < step < 100:
                            travelled += step
                            trail.append((cx, cy))
                    else:
                        trail.append((cx, cy))

                    selected_center = (cx, cy)

                    details.append(
                        (f"Travel: {length_text(travelled)}", color)
                    )

            # Draw distances between blue objects
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    p1 = centers[i]
                    p2 = centers[j]

                    cv2.line(
                        frame, p1, p2, (150, 150, 150), 1
                    )

                    distance = np.hypot(
                        p1[0] - p2[0],
                        p1[1] - p2[1]
                    )

                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2

                    put_text(
                        frame, length_text(distance),
                        mid_x, mid_y, (220, 220, 220), 0.4
                    )

            # Draw movement trail
            trail = trail[-100:]

            for i in range(1, len(trail)):
                cv2.line(
                    frame, trail[i - 1], trail[i],
                    (40, 40, 255), 2
                )

            # Show scale on camera view
            if pixels_per_cm is None:
                scale_label = "Scale not found - pixels only"
            else:
                scale_label = (
                    f"Scale: {pixels_per_cm:.1f} px/cm "
                    f"({scale_source})"
                )

            put_text(frame, scale_label, 10, 22)

            # Make the information panel
            panel = np.full(
                (480, 340, 3), (30, 30, 30), dtype=np.uint8
            )

            put_text(
                panel, "TASK 3 - OBJECT TRACKER",
                12, 28, (255, 255, 255), 0.55
            )
            cv2.line(
                panel, (12, 40), (328, 40), (90, 90, 90), 1
            )

            put_text(
                panel, f"Blue objects: {len(objects)}",
                12, 65, (255, 190, 90)
            )
            put_text(
                panel, scale_label,
                12, 90, (100, 220, 255), 0.42
            )

            if ball is None:
                put_text(
                    panel, "Tennis ball: not found",
                    12, 118, (160, 160, 160), 0.45
                )
            else:
                put_text(
                    panel, f"Ball centre: {ball['center']}",
                    12, 118, (80, 255, 100), 0.45
                )

            cv2.line(
                panel, (12, 135), (328, 135), (90, 90, 90), 1
            )

            y_pos = 158

            for text, color in details:
                if y_pos > 365:
                    put_text(
                        panel, "More details on camera view",
                        12, y_pos, (180, 180, 180), 0.38
                    )
                    break

                put_text(panel, text, 12, y_pos, color, 0.40)
                y_pos += 19

            # Show controls
            cv2.line(
                panel, (12, 385), (328, 385), (90, 90, 90), 1
            )
            put_text(
                panel, "CONTROLS", 12, 407,
                (255, 255, 255), 0.48
            )
            put_text(
                panel, "Left click: select object",
                12, 430, size=0.38
            )
            put_text(
                panel, "Right click: clear selection",
                12, 449, size=0.38
            )
            put_text(
                panel, "C: clear trail   M: masks   Q: quit",
                12, 468, size=0.36
            )

            # Show camera and panel together
            cv2.imshow(
                window, np.hstack((frame, panel))
            )

            # Show colour masks if requested
            if show_masks:
                cv2.imshow("Blue Mask", blue_mask)
                cv2.imshow("Ball Mask", ball_mask)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                trail = []
                travelled = 0.0
                selected_center = None

            elif key == ord("m"):
                show_masks = not show_masks

                if not show_masks:
                    cv2.destroyWindow("Blue Mask")
                    cv2.destroyWindow("Ball Mask")

    finally:
        # Close camera and windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
