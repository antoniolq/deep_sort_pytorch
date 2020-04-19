import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector_advanced
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

if __name__ == '__main__':
    test = build_detector_advanced()
    test()