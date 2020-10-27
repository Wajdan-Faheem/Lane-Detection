
import cv2
import numpy as np
import imutils
# Testing Lib
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('project_video.mp4')

'''
 ----------------------------------------
 Frame Measurement and adjustment script:
 ----------------------------------------
    Specific np.zeros mask for project_video
    Vertices = [(200,650),(530,475),(760,475),(1200,650)]


_, frame1 = cap.read()
frame2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
height = frame1.shape[0]
width = frame1.shape[1]
plt.imshow('frame',frame1)
plt.show()

h = frame1.shape[0]
w = frame1.shape[1]
vertices = [(0.156 * w, 0.902 * h), (0.414 * w, 0.659 * h), (0.593 * w, 0.659 * h), (0.937 * w, 0.902 * h)]
'''

Vertices = [(200,650),(530,475),(760,475),(1200,650)]

'''
-------------------------------
Perspective and Region Methods:
-------------------------------
'''

def changePerspective (inputFrame):
    
    frame_size = (inputFrame.shape[1],inputFrame.shape[0])
    inital_ROI = np.float32([[0.4375, 0.625], [.08, .9], [0.5703, 0.625], [0.94, .9]])
    end_ROI = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    inital_ROI = inital_ROI*np.float32(frame_size)
    end_RIOT = end_ROI*np.float32(frame_size)

    matrix_transformed = cv2.getPerspectiveTransform(inital_ROI,end_ROI)
    new_wrapped_frame = cv2.warpPerspective(inputFrame,matrix_transformed,frame_size)

    return new_wrapped_frame

def changePerspectiveBack (inputFrame):
    frame_size = (inputFrame.shape[1],inputFrame.shape[0])
    inital_ROI = np.float32([[0.4375, 0.625], [.08, .9], [0.5703, 0.625], [0.94, .9]])
    end_ROI = np.float32([[0, 0], [0, 1],[1, 0], [1, 1]])
    inital_ROI = inital_ROI*np.float32(frame_size)
    end_RIOT = end_ROI*np.float32(frame_size)

    matrix_transformed = cv2.getPerspectiveTransform(end_ROI,inital_ROI)
    new_wrapped_frame = cv2.warpPerspective(inputFrame,matrix_transformed,frame_size)

    return new_wrapped_frame

def ROI (inputFrame, vertices):
    zero_mask = np.zeros_like(inputFrame)  # Returns an black image the size of the original
    mask_colour = (255, 255, 255)
    cv2.fillPoly(zero_mask, vertices, mask_colour)
    roi = cv2.bitwise_and(inputFrame, zero_mask)
    return roi
 
'''
--------------------
Yellow Line Methods:
--------------------
'''

def Yellow_ColourFilter (inputFrame):
    
    hsv = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
    YellowLR = np.array([7, 60, 140])
    YellowHR = np.array([48, 255, 255])

    Yellow_Mask = cv2.inRange(hsv,YellowLR,YellowHR)
    return Yellow_Mask

def Yellow_canny(inputFrame):
    count =0
    Colour_Mask = Yellow_ColourFilter(inputFrame)
    canny_Image = cv2.Canny(Colour_Mask,50,150)

    while count < 5:
        count+=1
        canny = cv2.GaussianBlur(canny_Image,(5,5),0)

    return canny


def Yellow_DrawLines(inputFrame, yellowLines):
    copyFrame = np.copy(inputFrame)
    linesFrame = np.zeros_like(inputFrame)

    if np.any(yellowLines):
        for YellowLine in yellowLines:
            for x1,y1,x2,y2 in YellowLine:
                    locations = np.array([[x1,y1],[x2,y2]])
                    cv2.polylines(linesFrame,[locations], isClosed=False,color = (255,0,255),thickness =3)
        copyFrame = cv2.addWeighted(inputFrame,0.5,copyFrame,0.5,-400)
        mergedFrame = cv2.add(copyFrame,linesFrame)
        return mergedFrame
    else:
        copyFrame = np.copy(inputFrame) 
        mergedFrame = cv2.addWeighted(inputFrame,0.5,copyFrame,0.5,-400)

        return mergedFrame

'''
-------------------
White Line Methods:
-------------------
'''
def White_ColourFilter (inputFrame):
    hsv = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([220, 15, 255])

    White_Mask = cv2.inRange(hsv,lower_white,upper_white)
    return White_Mask

def White_canny(inputFrame):
    count = 0
    colour_Mask = White_ColourFilter(inputFrame)
    canny_Image = cv2.Canny(colour_Mask,50,150)
    
    while count < 5:
        count+=1
        canny = cv2.GaussianBlur(canny_Image,(5,5),0)
    return canny

def White_DrawLines(inputFrame, whiteLines):
    copyFrame = np.copy(inputFrame)
    linesFrame = np.zeros_like(inputFrame)

    if np.any(whiteLines):
        for YellowLine in whiteLines:
            for x1,y1,x2,y2 in YellowLine:
                    locations = np.array([[x1,y1],[x2,y2]])
                    cv2.polylines(linesFrame,[locations], isClosed=False,color = (0,0,255),thickness =3)
        copyFrame = cv2.addWeighted(inputFrame,0.5,copyFrame,0.5,-400)
        mergedFrame = cv2.add(copyFrame,linesFrame)
        return mergedFrame
    else:
        copyFrame = np.copy(inputFrame) 
        mergedFrame = cv2.addWeighted(inputFrame,0.5,copyFrame,0.5,-400)

        return mergedFrame
'''
--------------------
Car Centering Method:
--------------------
'''

def centerCar (inputFrame, Yellowlines,Whitelines,resultInital,width,height):
    # Copy the frame and write th lines to this given frame
    copyFrame = np.copy(inputFrame)
    lineFrame = np.zeros_like(inputFrame)
    # Origin = (650,720) for 1280X720 RES -> Ratio Provided below
    origin = (int(0.52*width), int(1*height))

    if np.any(Yellowlines) or np.any(Whitelines):
        direction_sensitivity = 0.12
        result = addResult(Yellowlines, Whitelines)
        if result <= resultInital[0]-np.array([direction_sensitivity*width]):
            arrow_Tip = np.array([result, int(0.60*height)], dtype=int)

            cv2.arrowedLine(lineFrame, origin, tuple(arrow_Tip), (0, 255, 255), 10)

            copyFrame = cv2.addWeighted(inputFrame, 0.5, copyFrame, 0.5, -400)
            mergedFrame = cv2.add(copyFrame, lineFrame)
            return mergedFrame, result

        elif result >= resultInital[0]+np.array([direction_sensitivity*width]):
            arrow_Tip = np.array([result, int(0.60*height)], dtype=int)
            cv2.arrowedLine(lineFrame, origin, tuple(arrow_Tip), (0, 255, 255), 10)

            copyFrame = cv2.addWeighted(inputFrame, 0.5, copyFrame, 0.5, -400)
            mergedFrame = cv2.add(copyFrame, lineFrame)
            return mergedFrame, result

        else:
            result = int(resultInital[0])
            arrow_Tip = np.array([result, int(0.60*height)])

            cv2.arrowedLine(lineFrame, origin, tuple(arrow_Tip), (0, 255, 255), 5)

            copyFrame = cv2.addWeighted(inputFrame, 0.5, copyFrame, 0.5, -400)
            mergedFrame = cv2.add(copyFrame, lineFrame)
            return mergedFrame, resultInital
    else:
        copyFrame = cv2.addWeighted(inputFrame, 0.5, copyFrame, 0.5, 0)
        return copyFrame, resultInital
'''
--------------------------
Basic Arithimetic Methods:
--------------------------
'''
def addResult (Yellow_lines, White_lines):
    Yellow_Line_Point = Yellow_Lines(Yellow_lines)
    White_line_Point = White_Lines(White_lines)
    result= (White_line_Point+Yellow_Line_Point)/2
    return result

def Yellow_Lines (Yellow_lines):
    if np.any(Yellow_lines):
        for Yellow_Line in Yellow_lines:
            for x1,y1,x2,y2 in Yellow_Line:
                    x2Value = np.array([x2])
        return(x2Value)
    else:
        return [0]

def White_Lines(White_lines):
    if np.any(White_lines):
        for White_Line in White_lines:
            for x1,y1,x2,y2 in White_Line:
                    x2Value = np.array([x2])
            return(x2Value)
    else:
        return[0]

'''
-------------
Main Methods:
-------------
'''

def main():

    resultInital = [0]

    while cap.isOpened():
        ret, frame = cap.read()

        # Image Resizing
        if frame is None:
            break
        frame = imutils.resize(frame, 1280)

        # Getting the vertices
        height = frame.shape[0]
        width = frame.shape[1]

        #Main body
        altered_Perspective = changePerspective(frame)
        Yellow_CannyImage = Yellow_canny(altered_Perspective)
        White_CannyImage = White_canny(altered_Perspective)

        Yellow_lines = cv2.HoughLinesP(Yellow_CannyImage,rho=6, theta=np.pi / 60, threshold=120, minLineLength=15, maxLineGap=30)
        White_lines = cv2.HoughLinesP(White_CannyImage,rho=6, theta=np.pi / 60, threshold=120, minLineLength=15, maxLineGap=30)
        
        Yellow_Lines_Frame = Yellow_DrawLines(altered_Perspective,Yellow_lines)
        White_Lines_Frame = White_DrawLines(altered_Perspective,White_lines)

        CenterLane, resultInital = centerCar(altered_Perspective, Yellow_lines, White_lines, resultInital, width, height)
        line_Frame = cv2.add(Yellow_Lines_Frame,White_Lines_Frame)
        New_Lines_frame= cv2.add(line_Frame,CenterLane)
        
        BackPerspective = changePerspectiveBack(New_Lines_frame)

        processedFrame = cv2.add(BackPerspective,frame)

        cv2.imshow('Final Image',processedFrame)
        # End condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ret:
            print('Video Was unable to open. Check file and location')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()