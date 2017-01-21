import numpy as np
import cv2

# from matplotlib import pyplot as plt

# cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)

# for i in range(10000):
#     # Capture frame-by-frame
#     ret, frame1 = cap1.read()
#     ret, frame2 = cap2.read()

#     # Our operations on the frame come here
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



#     stereo = cv2.StereoBM_create(numDisparities=10*16, blockSize=17)
#     disparity = stereo.compute(gray1,gray2)

#     # Display the resulting frame
#     cv2.imshow('frame1',gray1)
#     cv2.imshow('frame2',gray2)
#     #plt.imshow(disparity, 'gray')
#     #plt.imsave(str(i) + '.png', disparity, cmap="gray")
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()

#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
#from __future__ import print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        f.close()


if __name__ == '__main__':
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    print('loading images...')
    imgL = cv2.pyrDown( frame1 )  # downscale images for faster processing
    imgR = cv2.pyrDown( frame2 )

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 32
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 17,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.7*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
