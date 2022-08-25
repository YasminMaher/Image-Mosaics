import cv2 as cv
import sys
import matplotlib.pyplot as plt

import numpy as np

def convert(image):
    cv.imshow("Original Image", image)

    #grayscaled image
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("GS Image",grayImage)
    cv.waitKey(0)

    # apply  Median blur
    blurImage = cv.medianBlur(grayImage,5)
    cv.imshow("Blurred Image",blurImage)
    cv.waitKey(0)

    # detect Edges
    #Using Laplacian Filter
    ddepth=cv.CV_8U
    kernel_size=3

    edgeImage=cv.Laplacian(blurImage,ddepth,ksize=kernel_size)
    abs_edgeImage=cv.convertScaleAbs(edgeImage)
    cv.imshow("Edged Image", abs_edgeImage)
    cv.waitKey(0)

    th,BnWImage = cv.threshold(abs_edgeImage,10,255, cv.THRESH_BINARY_INV)
    cv.imshow("BNW Image",BnWImage)
    cv.waitKey(0)


    #painting the image using bilateral filter fastener

    #downsampling image to reduce resolution
    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(image.shape[1],image.shape[0])
    downsampled_img=cv.resize(image,dim)
    #cv.imshow("dsi Effect!", downsampled_img)

    # Painting the low resolution image using small bilateral filters
    filtered_img = downsampled_img
    for i in range(10):
        filtered_img = cv.bilateralFilter(filtered_img, 9, 9, 7)
    painted_image=filtered_img

    width2 = image.shape[1]
    height2 =image.shape[0]
    dim2 = (width2, height2)
    painted_image = cv.resize(painted_image, dim2)


    cv.imshow("Painted image", painted_image)


    # add together original image and edge image
    output = cv.bitwise_and( painted_image, painted_image, mask=BnWImage)

    cv.imshow("Cartoon Effect!", output)
    cv.waitKey(0)

img = cv.imread("C:/Users/Dell/PycharmProjects/Assignment1/TC.jpg")
convert(img)