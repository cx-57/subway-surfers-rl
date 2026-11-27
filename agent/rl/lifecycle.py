"""Game-over recognition uses explicit UI evidence, never missing player boxes."""

import cv2
import numpy as np


class GameOverDetector:
    def __init__(self, template, roi, threshold=0.94, confirmations=2, minimum_seconds=0.1):
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.roi = tuple(roi)
        self.threshold = float(threshold)
        self.confirmations = int(confirmations)
        self.minimum_seconds = float(minimum_seconds)
        if (self.template.size == 0 or self.template.std() < 5
                or len(self.roi) != 4 or not 0 <= self.roi[0] < self.roi[2] <= 1
                or not 0 <= self.roi[1] < self.roi[3] <= 1
                or not 0 < self.threshold <= 1 or self.confirmations < 1
                or not np.isfinite(self.minimum_seconds) or self.minimum_seconds < 0):
            raise ValueError("Invalid game-over template, ROI, or thresholds")
        self.reset()

    def reset(self):
        self.first_match = None
        self.count = 0

    def update(self, image, timestamp):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = image[round(y1 * height):round(y2 * height),
                     round(x1 * width):round(x2 * width)]
        if crop.size == 0:
            raise ValueError("Game-over ROI is empty")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.template.shape[1], self.template.shape[0]))
        error = np.abs(gray.astype(np.float32) - self.template.astype(np.float32)).mean()
        score = float(1.0 - error / 255.0)
        if score >= self.threshold:
            if self.first_match is None:
                self.first_match = timestamp
            self.count += 1
        else:
            self.reset()
        done = (self.count >= self.confirmations and self.first_match is not None
                and timestamp - self.first_match >= self.minimum_seconds)
        return bool(done), score
