#!/usr/bin/python

# TODO:
# Env -> frame stacking, image capture
# Agent -> convnet, memory, qval calculation
# loss, metrics, graphs

import cv2
import torch
import time
import pyautogui
import pyscreenshot as ps

gameover_screen = cv2.imread('gameover.jpg')

# gameover coord
# bbox=(910,920,1000,975)
# frame = ps.grab(bbox=(30,780,1915,1150))

# jump, nothing, duck


def get_frames(number): # TODO: skip
    frames = []

    for i in range(0, number):
        frame = ps.grab(bbox=(30,530,975,1475))
        frames.append(frame)

    return frames

    
def format_frames(frames):
    for frame in frames:
        frame_np = np.array(frame)
        old_size = frame_np.shape[:2]
        n_size = 64
        ratio = float(n_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        frame_res = cv2.resize(frame_np, (new_size[1], new_size[0]))

        # cv2.imshow('wtf2', frame_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gray_frame = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY) 

        return gray_frame


while True:

    frames = get_frames(4)

    is_gameover = (np.array(ps.grab(bbox=(910,920,1000,975))) == gameover_screen)[0][0][0] # TODO: fix


    time.sleep(4)
