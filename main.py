import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np
from numpy  import *
import random
import math

#Global Variables
global num_clicks



def img1_mousePoints(event,x,y,flags,param):
    global counter1
    if (num_clicks == counter1):
        return
    if event == cv2.EVENT_LBUTTONDOWN:#when you click the mouse
        circles1[counter1] = x,y #in row Cicles(0) put x0 and y0, in row cicles(1) put x1 and y1 and so on
        cv2.circle(img1, (circles1[counter1][0], circles1[counter1][1]),2, (0, 255, 0), cv2.FILLED)#color the point you've just clicked
        counter1 = counter1 + 1
        print(circles1)
def img2_mousePoints(event,x,y,flags,param):
    global counter2
    if (num_clicks == counter2):
        return
    if event == cv2.EVENT_LBUTTONDOWN:#when you click the mouse
        circles2[counter2] = x,y #in row Cicles(0) put x0 and y0, in row cicles(1) put x1 and y1 and so on
        cv2.circle(img2, (circles2[counter2][0], circles2[counter2][1]),2, (255, 0, 0), cv2.FILLED) #color the point you've just clicked
        counter2 = counter2 + 1
        print(circles2)
def img3_mousePoints(event,x,y,flags,param):
    global counter3
    if (num_clicks == counter3):
        return
    if event == cv2.EVENT_LBUTTONDOWN:#when you click the mouse
        circles3[counter3] = x,y #in row Cicles(0) put x0 and y0, in row cicles(1) put x1 and y1 and so on
        cv2.circle(img3, (circles3[counter3][0], circles3[counter3][1]),2, (255, 0, 0), cv2.FILLED) #color the point you've just clicked
        counter3 = counter3 + 1
        print(circles3)
def imgs_mousePoints(event,x,y,flags,param):
    global counter_s
    if (num_clicks == counter_s):
        return
    if event == cv2.EVENT_LBUTTONDOWN:#when you click the mouse
        circles_s[counter_s] = x,y #in row Cicles(0) put x0 and y0, in row cicles(1) put x1 and y1 and so on
        cv2.circle(Stitched_img, (circles_s[counter_s][0], circles_s[counter_s][1]),2, (255, 0, 0), cv2.FILLED) #color the point you've just clicked
        counter_s = counter_s + 1
        print(circles_s)


def FindCorespondances(First_img,Second_img,which_is_which):
    if(which_is_which==1):#the first project
        while True:  # this while is to colour the points

           cv2.imshow("first image", First_img)
           cv2.setMouseCallback("first image", img1_mousePoints)

           cv2.imshow("Second image", Second_img)
           cv2.setMouseCallback("Second image", img2_mousePoints)

           if ((counter1 == num_clicks) & (counter2 == num_clicks)):
                cv2.waitKey(0)
                break
           cv2.waitKey(1)

    if(which_is_which==2):#The second project 'bonus' Stitch 1
        # stitch the 2nd Image of Shanghai on the already stitched img
        while True:  # this while is to colour the points

            cv2.imshow("first image", First_img)
            cv2.setMouseCallback("first image", img1_mousePoints)

            cv2.imshow("Stitched image 1", Second_img)
            cv2.setMouseCallback("Stitched image 1", img3_mousePoints)

            if ((counter1 == num_clicks) & (counter3 == num_clicks)):
                cv2.waitKey(0)
                break
            cv2.waitKey(1)

    if(which_is_which==3):#The second project 'bonus' Final stitch
        # stitch the 2nd Image of Shanghai on the already stitched img
        while True:  # this while is to colour the points

            cv2.imshow("first image", First_img)
            cv2.setMouseCallback("first image", img2_mousePoints)

            cv2.imshow("Stitched image 1", Second_img)
            cv2.setMouseCallback("Stitched image 1", imgs_mousePoints)

            if ((counter2 == num_clicks) & (counter_s == num_clicks)):
                cv2.waitKey(0)
                break
            cv2.waitKey(1)

    return
def homography(circles1, circles2):
    print("circle1")
    print(circles1)
    print("circle2")
    print(circles2)
    #each correspondcy : x1,y1 //x2,y2  gives two equations, then we have 8 unknowns and last one would be=1
    H = np.zeros((3, 3))  # the homography matrix
    H.shape = (1, 9)  # there are 8 values unknown in this matrix , so we need Ax=B where x is the H

    # Create the A in Ax=Bwhich will be at least   # filling the matrix
    rows = num_clicks
    for row in range(rows):

        temp1 = np.zeros(9)
        temp2 = np.zeros(9)

        Xs = circles1[row][0]
        Ys = circles1[row][1]

        Xd = circles2[row][0]
        Yd = circles2[row][1]

        temp1[0] = -1 * Xs
        temp1[1] = -1 * Ys
        temp1[2] = -1

        # those are zeros
        # temp1[3]
        # temp1[4]
        # temp1[5]

        temp1[6] = Xs * Xd
        temp1[7] = Ys * Xd
        temp1[8] = Xd

        # those are zeros
        # temp2[0]
        # temp2[1]
        # temp2[2]

        temp2[3] = -1 * Xs
        temp2[4] = -1 * Ys
        temp2[5] = -1

        temp2[6] = Xs * Yd
        temp2[7] = Ys * Yd
        temp2[8] = Yd

        # print("temp1")
        # print(temp1)
        # print("temp2")
        # print(temp2)

        temp1 = np.asmatrix(temp1)
        temp2 = np.asmatrix(temp2)
        temp3 = np.concatenate((temp1, temp2))
        # print("temp3")
        # print(temp3)
        if (row == 0):
            the_stacked_array = temp3
            # print("the stacked array = ")
            # print(the_stacked_array)
        if (row > 0):
            the_stacked_array = np.concatenate((the_stacked_array, temp3))
            # print("the stacked array = ")
            # print(the_stacked_array)
    U, S, V = np.linalg.svd(the_stacked_array,full_matrices=True)
    # print(np.around(V,decimals=3))
    H=V[-1,:].reshape(3,3)
    print(np.around(H,decimals=3))
    return H
def Check_Homography(H,image_to_check,circles):
    # check if the homography matix is correctly computes
    coordinates = len(circles)
    print(coordinates)
    check = np.zeros((coordinates, 2), int)
    print(check)
    for coordinate in range(coordinates):
        src_coordinates = circles[coordinate]
        src_coordinates = np.asarray(src_coordinates)
        src_coordinates = src_coordinates.reshape(2, 1)
        z = matrix(1)
        src_coordinates = np.concatenate((src_coordinates, z))
        src_coordinates = np.asmatrix(src_coordinates)

        des_coordinates = np.matmul(H, src_coordinates)

        des_coordinates = des_coordinates / des_coordinates[2]
        check[coordinate][0] = des_coordinates[0]
        check[coordinate][1] = des_coordinates[1]
        cv2.circle(image_to_check, (check[coordinate][0], check[coordinate][1]), 4, (0, 0, 255),
                   cv2.FILLED)  # color the point you've just clicked

    print(check.astype('int32'))
    cv2.imshow("image2", image_to_check)
    cv2.waitKey(0)
    return

def IsOutSide(r,c,shape):
    if(c<0 or c>=shape[1]):
        return True
    if (r < 0 or r >= shape[0]):
        return True
    return False
def splatter(img_src1):
    Forward_warped_img = img_src1.copy()
    diameter =50
    for r in range(img_src1.shape[0]):
        for c in range(img_src1.shape[1]):
            r1= r + math.ceil(random.uniform(-0.5,0.5)*diameter)
            c1= c + math.ceil(random.uniform(-0.5,0.5)*diameter)

            if(IsOutSide(r1,c1,img_src1.shape)):
                Forward_warped_img.itemset((r,c,0),0)
                Forward_warped_img.itemset((r, c, 1), 0)
                Forward_warped_img.itemset((r, c, 2), 0)
            else:
                Forward_warped_img.itemset((r, c, 0), img_src1.item((r1,c1,0)))
                Forward_warped_img.itemset((r, c, 1), img_src1.item((r1, c1, 1)))
                Forward_warped_img.itemset((r, c, 2), img_src1.item((r1, c1, 2)))
    return Forward_warped_img
def Forward_warp(img1,H):

    Hight,Width,Channel = img1.shape
    created_img_coordiantes=[]
    x_values = []
    y_values = []
    print(Hight,Width)

    for y in range(Hight):
        for x in range(Width):
            pixel = np.array([[x,y,1]]).reshape(3,1)
            new_pixel = np.matmul(H,pixel)
            new_pixel[0][0] = new_pixel[0][0]/new_pixel[2][0]
            new_pixel[1][0] = new_pixel[1][0]/new_pixel[2][0]
            x_values.append(new_pixel[0][0])
            y_values.append(new_pixel[1][0])

    x_min, y_min = np.amin(x_values), np.amin(y_values)
    x_max, y_max = np.amax(x_values), np.amax(y_values)

    print(x_max,x_min)
    print(y_max,y_min)


    new_Width= int(x_max-x_min)
    x_values += -x_min

    new_Hight= int(y_max-y_min)
    y_values += -y_min


    print(x_max, x_min)
    print(y_max, y_min)

    print('New hight and width')

    print(new_Hight)
    print(new_Width)

    finalH= max(Hight,new_Hight)

    i = 0;

    Forward_warped_img=np.zeros((new_Hight,new_Width,3), np.uint8)
    averaging_weight = np.zeros(Forward_warped_img.shape, dtype=np.uint16)
    for y_src in range(Hight):
        for x_src in range(Width):
          if (int(y_values[i]) < new_Hight) and int(x_values[i]) < new_Width:
            Forward_warped_img[int(y_values[i]), int(x_values[i])] = img1[y_src, x_src]
            averaging_weight[int(y_values[i]),int(x_values[i])] +=1
            i += 1
    plt.imshow(cv2.cvtColor(Forward_warped_img,cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Final Forward Warped Image for bonus.jpg', Forward_warped_img)
    Forward_warped_img=splatter(Forward_warped_img)
    plt.imshow(cv2.cvtColor(Forward_warped_img, cv2.COLOR_BGR2RGB))
    plt.show()
    #cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Splattered Forward warp for project 1.jpg', Forward_warped_img)
    return Forward_warped_img,x_min,y_min

def Inverse_warp(Forward_warped_img,img_src1,H,x_min,y_min):

    H_inv = np.linalg.inv(H)  # get h inverse

    Warped_Hight, Warped_Width, Warped_Channel = Forward_warped_img.shape

    Inv_warped_img = np.zeros(Forward_warped_img.shape,np.uint8)

    Src_Hight, Src_Width, Src_Channel = img_src1.shape


    for c in range(Warped_Channel):
        for y in range(Warped_Hight):
            for x in range(Warped_Width):
                new_pixel = np.array([[x+x_min, y+y_min, 1]]).reshape(3, 1)
                pixel = np.matmul(H_inv, new_pixel)
                pixel[0][0] = pixel[0][0] / pixel[2][0]
                pixel[1][0] = pixel[1][0] / pixel[2][0]

                if pixel[0][0] == int(pixel[0][0]) and pixel[1][0] == int(pixel[1][0]) and (
                        pixel[0][0] < Src_Width and pixel[0][0] >= 0 and pixel[1][0] < Src_Hight and pixel[1][0] >= 0):
                    pixel[0][0] = int(pixel[0][0])
                    pixel[1][0] = int(pixel[1][0])
                    Inv_warped_img[y,x,c]= img_src1[pixel[1][0],pixel[0][0],c]
                else:#sub-pixeled
                    x_ceil = int(np.ceil(pixel[0][0]))
                    x_floor = int(np.floor(pixel[0][0]))
                    y_ceil = int(np.ceil(pixel[1][0]))
                    y_floor = int(np.floor(pixel[1][0]))
                    ptA = (x_ceil, y_floor)
                    ptB = (x_ceil, y_ceil)
                    ptC = (x_floor, y_floor)
                    ptD = (x_floor, y_ceil)
                    pts = [ptA, ptB, ptC, ptD]
                    for pt in pts:
                        if pt[0] >= Src_Width or pt[0] < 0 or pt[1] >= Src_Hight or pt[1] < 0:
                            continue

                       # ratio = (abs(pixel[0][0] - pt[0]) * abs(pixel[1][0] - pt[1]))/((x_ceil-x_floor)*(y_ceil-y_floor))

                        ratio = (abs(pixel[0][0] - pt[0]) * abs(pixel[1][0] - pt[1]))
                        intensity = ratio * img_src1[pt[1], pt[0], c]
                        Inv_warped_img[y,x,c] =Inv_warped_img[y,x,c] + intensity

    plt.imshow(cv2.cvtColor(Inv_warped_img, cv2.COLOR_BGR2RGB))
    plt.show()
#    cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Inverse warp for project 1.jpg', Inv_warped_img)
    return Inv_warped_img

def Stitch(Inv_warped_img,img_2,x_min,y_min):
    # img_2 = cv2.cvtColor((img_2,cv2.COLOR_BGR2RGB))
    Warped_Hight,Warped_Width,Warped_Channel=Inv_warped_img.shape

    Image_Hight,Image_Width,Image_Channel=img_2.shape

    print(Warped_Width,Warped_Hight)
    print(Image_Width, Image_Hight)
    Final_width=abs(Warped_Width-Image_Width)
    print(Final_width)
    Final_width=Warped_Width+int(x_min)
    print(Final_width)
    Stitched_Img=np.zeros((Warped_Hight,Final_width,3),np.uint8)


    for y in range(Warped_Hight):
        for x in range(Warped_Width):
            Stitched_Img[int(y), int(x+x_min)] = Inv_warped_img[int(y), x]

    plt.imshow(cv2.cvtColor(Stitched_Img,cv2.COLOR_BGR2RGB))
    plt.show()


    for y in range(Image_Hight):
        for x in range(Image_Width):
            Stitched_Img[int(y-y_min),int(x)] = img_2[y,x]
    plt.imshow(cv2.cvtColor(Stitched_Img,cv2.COLOR_BGR2RGB))
    plt.show()


    cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Stitched image for project 1.jpg', Stitched_Img)
    return
def StitchB(Inv_warped_img,img_2,x_min,y_min):
    # img_2 = cv2.cvtColor((img_2,cv2.COLOR_BGR2RGB))
    Warped_Hight,Warped_Width,Warped_Channel=Inv_warped_img.shape

    Image_Hight,Image_Width,Image_Channel=img_2.shape

    print(Warped_Width,Warped_Hight)
    print(Image_Width, Image_Hight)

    Final_width=abs(Warped_Width-Image_Width)
    print(Final_width)
    Final_width=Warped_Width+int(x_min)
    print(Final_width)

    Final_hight = abs(Warped_Hight - Image_Hight)
    print(Final_width)
    Final_hight = Warped_Hight + int(y_min)
    print(Final_hight)

    Stitched_Img=np.zeros((Final_hight
                           ,Final_width,3),np.uint8)


    for y in range(Warped_Hight):
        for x in range(Warped_Width):
            Stitched_Img[int(y+y_min), int(x+x_min)] = Inv_warped_img[int(y), x]

    plt.imshow(cv2.cvtColor(Stitched_Img,cv2.COLOR_BGR2RGB))
    plt.show()


    for y in range(Image_Hight):
        for x in range(Image_Width):
            Stitched_Img[int(y),int(x)] = img_2[y,x]
    plt.imshow(cv2.cvtColor(Stitched_Img,cv2.COLOR_BGR2RGB))
    plt.show()


    return Stitched_Img

def Project1():
    global num_clicks
    global circles1
    global circles2

    '''BEGIN 1'''
    '''INITIALISATIONS'''

    yes_or_no = input("If you want to mouse click on edges press 1 \nelse press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        num_clicks = input("How many clicks you'll make : ")
        num_clicks = int(num_clicks)

        counter1 = 0  # how many clicks
        counter2 = 0  # how many clicks

        circles1 = np.zeros((num_clicks, 2), int)
        circles2 = np.zeros((num_clicks, 2), int)

        # first function: we get correspondecies
        FindCorespondances(img1,img2,1)
    else:
        num_clicks=25
        # without clicks
        circles1 = [[12, 178],
                    [50, 169],
                    [52, 395],
                    [78, 385],
                    [72, 495],
                    [93, 495],
                    [80, 502],
                    [102, 503],
                    [100, 522],
                    [80, 521],
                    [82, 591],
                    [76, 615],
                    [102, 615],
                    [250, 609],
                    [302, 539],
                    [390, 559],
                    [390, 574],
                    [443, 558],
                    [441, 575],
                    [370, 292],
                    [438, 303],
                    [475, 308],
                    [498, 323],
                    [522, 315],
                    [526, 336]]
        circles2 = [[465, 239],
                    [496, 230],
                    [515, 444],
                    [536, 429],
                    [535, 535],
                    [556, 537],
                    [542, 542],
                    [564, 543],
                    [564, 559],
                    [542, 558],
                    [551, 627],
                    [546, 650],
                    [572, 647],
                    [715, 645],
                    [759, 574],
                    [851, 593],
                    [852, 608],
                    [905, 593],
                    [907, 611],
                    [813, 320],
                    [883, 327],
                    [926, 328],
                    [948, 342],
                    [976, 332],
                    [979, 354]]

    H12 = homography(circles1, circles2)
    image_to_check = img_src2.copy()

    Check_Homography(H12, image_to_check, circles1)

    yes_or_no = input("If you want to use forward warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Forward_warped_img, x_min, y_min = Forward_warp(img_src1, H12)
    else:
        # Without forwarding
        Forward_warped_img=cv2.imread('Forward_warped_img.jpg')
        x_min=444.48359946849655 #for first proj
        y_min=-89.93914157462093 # for first proj

    yes_or_no = input("If you want to use inverse warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Inverse_warped_img = Inverse_warp(Forward_warped_img, img_src1, H12, x_min, y_min)
    else:
        # Without Inversing
        Inverse_warped_img=cv2.imread('Inverse warp for project 1.jpg')


    plt.imshow(cv2.cvtColor(Inverse_warped_img, cv2.COLOR_BGR2RGB))
    plt.show()
    Stitch(Inverse_warped_img, img_src2, x_min, y_min)
    return
def Bonus():
    '''BEGIN 2'''
    global num_clicks
    global circles1
    global circles2
    global circles3
    global circles_s

    '''First Stitch Bonus'''

    yes_or_no = input("If you want to mouse click on edges press 1 \nelse press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        num_clicks = input("How many clicks you'll make : ")
        num_clicks = int(num_clicks)

        circles1 = np.zeros((num_clicks, 2), int)
        circles3 = np.zeros((num_clicks, 2), int)

        counter1 = 0  # how many clicks
        counter3 = 0  # how many clicks

        FindCorespondances(img1, img3, 2)
    else:
        num_clicks=20
        # without mouse clicks
        circles3=[[648, 182],
         [639, 168],
         [596, 257],
         [584, 328],
         [594, 311],
         [658, 469],
         [122,  75],
         [136,  80],
         [125,  16],
         [141,  18],
         [ 99, 146],
         [155, 148],
         [ 19, 371],
         [ 40, 309],
         [  6, 458],
         [178, 455],
         [305, 415],
         [185, 521],
         [293, 376],
         [471, 316]]
        circles1=[[ 992 , 175],
         [ 982 , 163],
         [ 936 , 256],
         [ 927 , 330],
         [ 940 , 311],
         [1012 , 474],
         [ 453 , 103],
         [ 468  ,106],
         [ 455  , 47],
         [ 469  , 48],
         [ 434  ,172],
         [ 488  ,170],
         [ 367  ,394],
         [ 385  ,334],
         [ 361  ,478],
         [ 523  ,470],
         [ 645  ,428],
         [ 530  ,536],
         [ 633  ,393],
         [ 809  ,322]]


    H31 = homography(circles3, circles1)
    print(H31)

    image_to_check = img_src1
    Check_Homography(H31, image_to_check, circles3)

    yes_or_no = input("If you want to use forward warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Forward_warped_img, x_min, y_min = Forward_warp(img_src3, H31)
        cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/First Forward Warped Image for bonus.jpg',Forward_warped_img)

    else:
        # without forwarding
        Forward_warped_img=cv2.imread('First Forward Warped Image for bonus.jpg')
        x_min=336.777196087556
        y_min=-47.42372568452973

    yes_or_no = input("If you want to use inverse warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Inverse_warped_img = Inverse_warp(Forward_warped_img, img_src3, H31, x_min, y_min)
        cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/First Inverse Warped Image for Bonus.jpg', Inverse_warped_img)
    else:
        # without inverse
        Inverse_warped_img=cv2.imread('First Inverse Warped Image for Bonus.jpg')


    plt.imshow(cv2.cvtColor(Inverse_warped_img, cv2.COLOR_BGR2RGB))
    plt.show()
    Stitched_Img=StitchB(Inverse_warped_img, img1, x_min, y_min)
    cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Stitched_Img2_bonus.jpg', Stitched_Img)

    ''' Final Stitch in Bonus '''

    yes_or_no = input("If you want to mouse click on edges press 1 \nelse press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        num_clicks = input("How many clicks you'll make : ")
        num_clicks = int(num_clicks)

        counter2 = 0  # how many clicks
        counter_s = 0  # how many clicks

        circles2 = np.zeros((num_clicks, 2), int)
        circles_s = np.zeros((num_clicks, 2), int)

        FindCorespondances(img2, Stitched_img, 3)
    else:
        num_clicks=20
        # without mouse clicks
        circles_s=[[453,  46],
         [468 , 47],
         [454, 106],
         [467, 107],
         [434, 174],
         [489, 175],
         [522, 472],
         [463, 470],
         [385, 334],
         [654, 464],
         [646, 426],
         [360, 478],
         [173, 435],
         [242, 341],
         [808, 322],
         [889, 374],
         [936, 256],
         [983, 163],
         [904, 475],
         [552, 310]]
        circles2=[[ 485,  245],
         [ 498,  246],
         [ 484,  303],
         [ 498,  305],
         [ 462,  369],
         [ 518,  370],
         [ 553,  671],
         [ 490 , 669],
         [ 412 , 527],
         [ 684 , 664],
         [ 676 , 624],
         [ 385 , 676],
         [ 197 , 627],
         [ 270 , 534],
         [ 839 , 519],
         [ 921 , 572],
         [ 966 , 451],
         [1009 , 360],
         [ 939 , 675],
         [ 581 , 506]]

    HS2 = homography(circles_s, circles2)
    print(HS2)

    image_to_check = cv2.imread('shanghai-22.png')
    Check_Homography(HS2, image_to_check, circles_s)

    yes_or_no = input("If you want to use forward warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Forward_warped_img, x_min, y_min = Forward_warp(Stitched_img_src, HS2)
        cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Splattered Forward warp for Bonus.jpg',Forward_warped_img)
    else:
        # without forward warping
        Forward_warped_img=cv2.imread('Splattered Forward warp for Bonus.jpg')
        x_min=9.658557738438127
        y_min= 199.99825628575684

    yes_or_no = input("If you want to use inverse warping press 1 \nIf you want to use the already saved photo press 0 : ")
    yes_or_no = int(yes_or_no)
    if(yes_or_no==1):
        Inverse_warped_img = Inverse_warp(Forward_warped_img, Stitched_img_src, HS2, x_min, y_min)
        #cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Final Inverse Warped Image for bonus.jpg', Inverse_warped_img)

    else:
        # without Inverse warp
        Inverse_warped_img=cv2.imread('Final_Inverse_warped_img2_bonus.jpg')


    plt.imshow(cv2.cvtColor(Inverse_warped_img, cv2.COLOR_BGR2RGB))
    plt.show()
    Stitched_Img=StitchB(Inverse_warped_img, img_src2, x_min, y_min)
    cv2.imwrite('C:/Users/Dell/PycharmProjects/Mosaics/Final Stitched Image for bonus.jpg', Stitched_Img)

    return




'''INITIALISATIONS'''
img_src1 = cv2.imread('image1.jpg')
img_src2 = cv2.imread('image2.jpg')
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

counter1 = 0  # how many clicks
counter2 = 0  # how many clicks
Project1()



'''INITIALISATIONS'''
img_src1 = cv2.imread('shanghai-21.png')
img_src2 = cv2.imread('shanghai-22.png')
img_src3 = cv2.imread('shanghai-23.png')
Stitched_img_src = cv2.imread('Stitched_Img2_bonus.jpg')

img1 = cv2.imread('shanghai-21.png')
img2 = cv2.imread('shanghai-22.png')
img3 = cv2.imread('shanghai-23.png')
Stitched_img = cv2.imread('Stitched_Img2_bonus.jpg')

counter1 = 0  # how many clicks
counter3 = 0  # how many clicks
counter2 = 0  # how many clicks
counter_s = 0  # how many clicks
Bonus()
exit()
