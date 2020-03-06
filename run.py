import cv2
import numpy as np
import os
import subprocess
import glob
from matplotlib import pyplot as plt

DEBUG=True

def plot_image(image, title):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.show()


def find_street_lanes(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    canny_image = cv2.Canny(blur_image, 100, 150)
    padded_image = padding(canny_image)
    hough_lines_image = hough_lines(padded_image)
    combined_image = combine_image(hough_lines_image, image)
    return combined_image


def padding(image):
    bottom_padding = 100
    height = image.shape[0]
    width = image.shape[1]
    bottom_left = [0, height - bottom_padding]
    bottom_right = [width, height - bottom_padding]
    top_right = [width * 1 / 3, height * 1 / 3]
    top_left = [width * 2 / 3, height * 1 / 3]
    vertices = [np.array([bottom_left, bottom_right, top_left, top_right], dtype=np.int32)]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    padded_image = cv2.bitwise_and(image, mask)

    return padded_image


def averaged_lines(image, lines):
    right_lines = []
    left_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope >= 0:
            right_lines.append([slope, intercept])
        else:
            left_lines.append([slope, intercept])

    def merge_lines(image, lines):
        if len(lines) > 0:
            slope, intercept = np.average(lines, axis=0)
            y1 = image.shape[0]
            y2 = int(y1 * (1 / 2))
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return np.array([x1, y1, x2, y2])

    left = merge_lines(image, left_lines)
    right = merge_lines(image, right_lines)
    return left, right


def hough_lines(image, rho=0.9, theta=np.pi / 180, threshold=100, min_line_len=100, max_line_gap=50):
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is not None:
        lines = averaged_lines(image, lines)
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 20)
    return lines_image

def combine_image(image, ini_image, alpha=0.9, beta=1.0, lamb=0.0):
    combined_image = cv2.addWeighted(ini_image, alpha, image, beta, lamb)
    return combined_image

if __name__ == '__main__':
    answer = input("Video (1) or Image (2)?: ")
    if answer == "1":
        test_video = cv2.VideoCapture("input/test_video.mov")
        while True:
            _, frame = test_video.read()
            if frame is not None:
                street_lane_detected = find_street_lanes(frame)
                cv2.imshow("Video", street_lane_detected)
                cv2.waitKey(20)
            else:
                break
        test_video.release()
        cv2.destroyAllWindows()
        exit()
    elif answer == "2":
        test_image = cv2.imread("input/test_image.jpg")
        street_lane_detected = street_lanes = find_street_lanes(test_image)
        cv2.imshow('resulting image', street_lane_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()




