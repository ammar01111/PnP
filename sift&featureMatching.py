import cv2

#Reading Images
img1 = cv2.imread('C:/Users/PC/Desktop/Image Dataset/ImageDataset/01.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/Users/PC/Desktop/Image Dataset/ImageDataset/02.png',cv2.IMREAD_GRAYSCALE)

#Scaling the read Images
scaledPer = 20
width1 = int(img1.shape[1] *scaledPer/100)
height1 = int(img1.shape[0] *scaledPer/100)

width2 = int(img2.shape[1] *scaledPer/100)
height2 = int(img2.shape[0] *scaledPer/100)
resizedImage1 = cv2.resize(img1,(width1,height1), interpolation= cv2.INTER_AREA)
resizedImage2 = cv2.resize(img2,(width2,height2), interpolation= cv2.INTER_AREA)


#SIFT
sift = cv2.SIFT_create()

#Detecting keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

###img1
##For cirles as keypoints
#circles1 = cv2.drawKeypoints(img1,kp1,img1,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
#resizedCircle1 = cv2.resize(circles1,(width1,height1), interpolation= cv2.INTER_AREA)
#For points as keypoints
points1 = cv2.drawKeypoints(img1,kp1,img1,(0,0,255),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
resizedPoints1 = cv2.resize(points1,(width1,height1), interpolation= cv2.INTER_AREA)
##For keypoints with size and orientation
#rich1 = cv2.drawKeypoints(img1,kp1,img1,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#resizedRich1 = cv2.resize(rich1,(width1,height1), interpolation= cv2.INTER_AREA)

###img2
##For cirles as keypoints
#circles2 = cv2.drawKeypoints(img2,kp2,img2,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
#resizedCircle2 = cv2.resize(circles2,(width2,height2), interpolation= cv2.INTER_AREA)
#For points as keypoints
points2 = cv2.drawKeypoints(img2,kp2,img2,(0,0,255),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
resizedPoints2 = cv2.resize(points2,(width2,height2), interpolation= cv2.INTER_AREA)
##For keypoints with size and orientation
#rich2 = cv2.drawKeypoints(img2,kp2,img2,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#resizedRich2 = cv2.resize(rich2,(width2,height2), interpolation= cv2.INTER_AREA)


#Showing the interpolated Images
#cv2.imshow('img1', resizedImage1)
#cv2.imshow('Circles', resizedCircle1)
cv2.imshow('Points1', resizedPoints1)
#cv2.imshow('Rich', resizedRich1)

#cv2.imshow('img2', resizedImage2)
#cv2.imshow('Circles', resizedCircle2)
cv2.imshow('Points2', resizedPoints2)
#cv2.imshow('Rich', resizedRich2)

#Feature Matching using Brute-Force Method
bf = cv2.BFMatcher()
Matches = bf.knnMatch(des1,des2,2) #For two nearest Values

#For minimizing matching errors 
goodMatches = []
for m,n in Matches:
    if m.distance < 0.75*n.distance:
        goodMatches.append([m])

matchedImage = cv2.drawMatchesKnn(img1,kp1,img2,kp2,goodMatches,None,(255,0,0),(0,255,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('Match1.png',matchedImage)

#To view the image
#scaledMatchedImg = cv2.resize(matchedImage,(matchedImage.shape[1]*scaledPer/100,matchedImage.shape[0]*scaledPer/100),interpolation= cv2.INTER_AREA)
#cv2.imshow('Matched Image',scaledMatchedImg)

key = cv2.waitKey(0) 
if cv2.waitKey == ord('q'):
    print("Closing the windows")
    cv2.destroyAllWindows()
