#!/usr/bin/env python
import cv2 as cv
import numpy as np


CAMERA_MATRIX = np.array([
    [983.91572635,   0.0, 480.52646536],   # fx, 0, cx
    [  0.0, 984.66144487, 651.75676875],   # 0, fy, cy
    [  0.0,   0.0,   1.0]
])
DIST_COEFS = np.array([0.1, -0.2, 0.0, 0.0, 0.0])

FX = CAMERA_MATRIX[0, 0]
FY = CAMERA_MATRIX[1, 1]

CAMERA_HEIGHT_CM = 20.0

MIN_CONTOUR_AREA = 800
CANNY_LOW = 50
CANNY_HIGH = 150


def undistort_frame(frame, new_camera_matrix):
    return cv.undistort(frame, CAMERA_MATRIX, DIST_COEFS, None, new_camera_matrix)


def detect_objects(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edged = cv.dilate(edged, None, iterations=2)
    edged = cv.erode(edged, None, iterations=1)

    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv.contourArea(c) > MIN_CONTOUR_AREA]
    return valid_contours


def pixels_to_cm(pixel_length, focal_px):
    return (pixel_length * CAMERA_HEIGHT_CM) / focal_px


def main():
    cap = cv.VideoCapture(1)
    ret, frame = cap.read()
    if not ret:
        print("ERORR!!")
        return

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        CAMERA_MATRIX, DIST_COEFS, (w, h), 1, (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = undistort_frame(frame, new_camera_matrix)
        contours = detect_objects(frame)

        for c in contours:
            rect = cv.minAreaRect(c)
            (cx, cy), (pw, ph), angle = rect

            width_px, height_px = max(pw, ph), min(pw, ph)
            real_width_cm = pixels_to_cm(width_px, FX)
            real_height_cm = pixels_to_cm(height_px, FY)

            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame, [box], -1, (0, 255, 0), 2)

            (tl, tr, br, bl) = box

            top_mid_x = int((tl[0] + tr[0]) / 2)
            top_mid_y = int((tl[1] + tr[1]) / 2)

            left_mid_x = int((tl[0] + bl[0]) / 2)
            left_mid_y = int((tl[1] + bl[1]) / 2)

            FONT_SCALE = 0.8
            THICKNESS = 2
            TEXT_COLOR = (0,0,255)

            cv.putText(frame, f"{real_width_cm:.1f} cm", (top_mid_x - 40, top_mid_y - 15),
                       cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

            cv.putText(frame, f"{real_height_cm:.1f} cm", (left_mid_x - 70, left_mid_y),
                       cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

        cv.imshow("Object Dimensions (Live)", frame)
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()