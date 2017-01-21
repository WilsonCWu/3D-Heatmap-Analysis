import cv2
import numpy as np

SIDE_SIZE = 150

FRAMES_IN = 0
FRAMES_OUT = 0
FRAMES_LENGTH = 3

PEOPLE_IN_FRAME = 0

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_flow(img, flow, step=16):
    threshold = 1.5
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    avg_x = np.average(fx, axis=0)
    avg_y = np.average(fy, axis=0)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    if avg_x > threshold:
        return vis, 1
    elif avg_x < -threshold:
        return vis, -1
    return vis, 0

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

cam = cv2.VideoCapture(1)

winName = "Movement detection"
#cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

last = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
this = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
img = cam.read()[1]
first = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

prevgrayLeft = cv2.cvtColor(img[0:, -SIDE_SIZE:], cv2.COLOR_BGR2GRAY)
prevgrayRight = cv2.cvtColor(img[0:, :SIDE_SIZE], cv2.COLOR_BGR2GRAY)

while True:
    # Checking for movement

    avg = np.average(np.average(diffImg(last, this, first), axis=0), axis=0)
    if avg >= 1:
        print("I see you!")
    cv2.imshow( winName, diffImg(last, this, first) )

    last = this
    this = first
    img = cam.read()[1]
    first = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Looking for hoomans

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    try:
        if img is None:
            print('Failed to load image file:', fn)
            continue
    except:
        print('loading error')
        continue

    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    #print('%d (%d) found' % (len(found_filtered), len(found)))
    cv2.imshow('img', img)

    # Trying to do ROI

    rightEdge = img[0: , :SIDE_SIZE]
    leftEdge = img[0: , -SIDE_SIZE:]

    # Checking for movement direction

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    cv2.imshow('flow', draw_flow(gray, flow)[0])

    grayLeft = cv2.cvtColor(leftEdge, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(rightEdge, cv2.COLOR_BGR2GRAY)

    flowRight = cv2.calcOpticalFlowFarneback(prevgrayRight, grayRight, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flowLeft  = cv2.calcOpticalFlowFarneback(prevgrayLeft, grayLeft, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    prevgrayRight = grayRight
    prevgrayLeft = grayLeft

    leftImg, moveLeft = draw_flow(grayLeft, flowLeft)
    rightImg, moveRight = draw_flow(grayRight, flowRight)

    print(moveRight, moveLeft)

    if moveLeft == -1:
        print("moving left on left side")
        FRAMES_IN += 1
    elif moveLeft == 1:
        print("moving right on right side")
        FRAMES_OUT += 1
    if moveRight == 1:
        print("moving right on right side")
        FRAMES_IN += 1
    elif moveRight == -1:
        print("moving left on right side")
        FRAMES_OUT += 1

    cv2.imshow("left edge dir",  rightImg)
    cv2.imshow("right edge dir", leftImg)

    PEOPLE_IN_FRAME += FRAMES_IN // FRAMES_LENGTH - FRAMES_OUT // FRAMES_LENGTH
    print(PEOPLE_IN_FRAME)
    FRAMES_IN %= FRAMES_LENGTH
    FRAMES_OUT %= FRAMES_LENGTH


    ch = cv2.waitKey(5)
    if ch == 27:
        break

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break