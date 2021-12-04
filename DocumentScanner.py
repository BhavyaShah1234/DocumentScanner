import cv2 as cv
import numpy as np


def nothing(a):
    print(a)


def blur_image(image, tune, k1_b, sx_b, sy_b):
    if tune:
        k1 = cv.getTrackbarPos('Kernel1', 'Blur')
        sx = cv.getTrackbarPos('SigmaX', 'Blur')
        sy = cv.getTrackbarPos('SigmaY', 'Blur')
        if k1 % 2 == 0:
            k1 = k1 + 1
    else:
        k1 = k1_b
        sx = sx_b
        sy = sy_b
        if k1 % 2 == 0:
            k1 = k1 + 1
    image = cv.GaussianBlur(image, ksize=(k1, k1), sigmaX=sx, sigmaY=sy)
    return image


def canny_image(image, tune, t1_b, t2_b):
    if tune:
        t1 = cv.getTrackbarPos('Threshold1', 'Canny')
        t2 = cv.getTrackbarPos('Threshold2', 'Canny')
    else:
        t1 = t1_b
        t2 = t2_b
    image = cv.Canny(image, threshold1=t1, threshold2=t2)
    return image


def dilate_erode_image(image, tune, k2_b, dilate_iter_b, erode_iter_b):
    if tune:
        k2 = cv.getTrackbarPos('Kernel2', 'Dilate Erode')
        dilate_iter = cv.getTrackbarPos('Dilate', 'Dilate Erode')
        erode_iter = cv.getTrackbarPos('Erode', 'Dilate Erode')
        if k2 % 2 == 0:
            k2 = k2 + 1
    else:
        k2 = k2_b
        dilate_iter = dilate_iter_b
        erode_iter = erode_iter_b
        if k2 % 2 == 0:
            k2 = k2 + 1
    kernel = np.ones(shape=(k2, k2))
    dilate = cv.dilate(image, kernel=kernel, iterations=dilate_iter)
    erode = cv.erode(dilate, kernel=kernel, iterations=erode_iter)
    return erode


def detect_contours(input_image, display_image, draw, threshold_area):
    contours, _ = cv.findContours(input_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_list = list(contours)
    for contour in contours:
        area = cv.contourArea(contour)
        if area < threshold_area:
            contours_list.remove(contour)
        else:
            if draw:
                display_image = cv.drawContours(display_image, contour, -1, (0, 0, 0), 3)
    return contours_list, display_image


def get_corner_points(contours_list, display_image, draw):
    corners = []
    for contour in contours_list:
        perimeter = cv.arcLength(contour, True)
        corner = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        corners.append([[corner[0][0][0], corner[0][0][1]],
                        [corner[1][0][0], corner[1][0][1]],
                        [corner[2][0][0], corner[2][0][1]],
                        [corner[3][0][0], corner[3][0][1]]])
        if draw:
            display_image = cv.circle(display_image, (corner[0][0][0], corner[0][0][1]), 2, (255, 0, 0), 3)
            display_image = cv.circle(display_image, (corner[1][0][0], corner[1][0][1]), 2, (0, 255, 0), 3)
            display_image = cv.circle(display_image, (corner[2][0][0], corner[2][0][1]), 2, (0, 0, 255), 3)
            display_image = cv.circle(display_image, (corner[3][0][0], corner[3][0][1]), 2, (0, 255, 255), 3)
    return corners, display_image


def warp_image(corners):
    warps = []
    for obj in corners:
        p1, p2, p3, p4 = obj
        w = int(pow(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2), 0.5))
        h = int(pow(pow(p2[0] - p3[0], 2) + pow(p2[1] - p3[1], 2), 0.5))
        pts1 = np.array([p2, p1, p3, p4], dtype='float32')
        pts2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        warped = cv.warpPerspective(img, matrix, (w, h))
        if w > h:
            warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
        warps.append(warped)
    return warps


img_path = 'c3.jpg'
fine_tuning = False
draw_helper = False
draw_contours = False
draw_corners = True
draw_warp = True
draw_cards = True

k1_best = 3
sx_best = 2
sy_best = 2
t1_best = 95
t2_best = 300
k2_best = 3
dilate_iter_best = 3
erode_iter_best = 2
min_area = 10000

if fine_tuning:
    cv.namedWindow('Blur')
    cv.createTrackbar('Kernel1', 'Blur', 1, 10, nothing)
    cv.createTrackbar('SigmaX', 'Blur', 1, 10, nothing)
    cv.createTrackbar('SigmaY', 'Blur', 1, 10, nothing)
    cv.namedWindow('Canny')
    cv.createTrackbar('Threshold1', 'Canny', 1, 300, nothing)
    cv.createTrackbar('Threshold2', 'Canny', 1, 300, nothing)
    cv.namedWindow('Dilate Erode')
    cv.createTrackbar('Kernel2', 'Dilate Erode', 1, 10, nothing)
    cv.createTrackbar('Dilate', 'Dilate Erode', 1, 10, nothing)
    cv.createTrackbar('Erode', 'Dilate Erode', 1, 10, nothing)

while True:
    img = cv.imread(img_path)
    img = cv.resize(img, (None, None), None, fx=0.2, fy=0.2)
    img_copy1 = img.copy()
    img_copy2 = img.copy()
    img_copy3 = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = blur_image(gray, fine_tuning, k1_best, sx_best, sy_best)
    canny = canny_image(blur, fine_tuning, t1_best, t2_best)
    dilate_erode = dilate_erode_image(canny, fine_tuning, k2_best, dilate_iter_best, erode_iter_best)
    object_contours, contour_img = detect_contours(dilate_erode, img_copy1, draw_contours, min_area)
    corner_points, corner_img = get_corner_points(object_contours, img_copy2, draw_corners)
    warp_list = warp_image(corner_points)

    if draw_helper:
        cv.imshow('Image', img)
        cv.imshow('Gray', gray)
        cv.imshow('Blur', blur)
        cv.imshow('Canny', canny)
        cv.imshow('Dilate Erode', dilate_erode)

    if draw_contours:
        cv.imshow('Contours', contour_img)

    if draw_corners:
        cv.imshow('Corners', corner_img)

    if draw_corners:
        for index, warp_img in enumerate(warp_list):
            cv.imshow(f'Cards{index}', warp_img)

    key = cv.waitKey(1)
    if key == 113:
        break
