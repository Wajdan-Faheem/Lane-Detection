#Region Of Interest (ROI)
import cv2 as cv
import numpy as np
Vertices = [(200,650),(530,475),(760,475),(1200,650)]

cap = cv.VideoCapture('project_video.mp4')

def ROI(inputFrame,Vertices):
    zero_mask = np.zeros_like(inputFrame)
    mask_color =(255,255,255)
    cv.fillPoly(zero_mask,Vertices, mask_color)
    roi = cv.bitwise_and(inputFrame,zero_mask)
    return roi

def main():
    while(cap.isOpened()):
        ret,frame = cap.read()
        
        ROI_View = ROI(frame,np.array([Vertices]))

        cv.imshow('Input Frame', frame) 
        cv.imshow('ROI', ROI_View)
        k=cv.waitKey(1)
        if k ==ord("q"):
            break
        if not ret:
            print("Vide was not available")
    cap.release()
    cv.destroyAllWindows()


if __name__ ==("__main__"):
    main()