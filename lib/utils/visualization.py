import numpy as np
import cv2
_WHITE = (255, 255, 255)
def draw_skeleton(aa, kp, show_skeleton_labels=False):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    for i, j in skeleton:
        if kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1 and kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0:
            cv2.line(aa, tuple(kp[i-1][:2]), tuple(kp[j-1][:2]), (0,255,255), 2)
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:
            if kp[j][2] > 1.1:
                cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((0,0,255)), 2)
            elif kp[j][2] > 0.1:
                cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((255,0,0)), 2)

            if show_skeleton_labels and kp[j][2] > 0.1:
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
    
