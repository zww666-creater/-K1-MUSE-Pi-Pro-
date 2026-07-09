#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import cv2
import subprocess

print("========== camera probe ==========")

print("\n[1] /dev/video*:")
for p in sorted(glob.glob("/dev/video*")):
    try:
        st = os.stat(p)
        print(f"  {p} mode={oct(st.st_mode)}")
    except Exception as e:
        print(f"  {p} stat failed: {e}")

print("\n[2] fuser:")
subprocess.run("fuser -v /dev/video* 2>/dev/null || true", shell=True)

print("\n[3] v4l2-ctl --list-devices:")
subprocess.run("v4l2-ctl --list-devices 2>/dev/null || echo 'v4l2-ctl not found'", shell=True)

print("\n[4] v4l2-ctl --all:")
for p in sorted(glob.glob("/dev/video*")):
    print(f"\n----- {p} -----")
    subprocess.run(f"v4l2-ctl -d {p} --all 2>/dev/null | sed -n '1,80p' || true", shell=True)

print("\n[5] OpenCV build info:")
info = cv2.getBuildInformation()
for line in info.splitlines():
    if "Video I/O" in line or "V4L" in line or "GStreamer" in line or "FFMPEG" in line:
        print(line)

print("\n[6] OpenCV open test:")

backends = [
    ("CAP_V4L2", cv2.CAP_V4L2),
    ("CAP_ANY", cv2.CAP_ANY),
]

fourccs = [
    None,
    "MJPG",
    "YUYV",
]

devices = []

for p in sorted(glob.glob("/dev/video*")):
    devices.append(p)
    try:
        idx = int(p.replace("/dev/video", ""))
        devices.append(idx)
    except Exception:
        pass

seen = set()
unique_devices = []
for d in devices:
    key = str(d)
    if key not in seen:
        seen.add(key)
        unique_devices.append(d)

for dev in unique_devices:
    for backend_name, backend in backends:
        for fourcc in fourccs:
            label = f"dev={dev}, backend={backend_name}, fourcc={fourcc}"

            try:
                cap = cv2.VideoCapture(dev, backend)

                if fourcc is not None:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)

                opened = cap.isOpened()

                ok = False
                shape = None

                if opened:
                    for _ in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            ok = True
                            shape = frame.shape
                            break
                        time.sleep(0.05)

                cap.release()

                print(f"[TEST] {label} -> opened={opened}, read={ok}, shape={shape}")

                if ok:
                    print(f"[SUCCESS] 可用摄像头配置: {label}")
                    raise SystemExit(0)

            except SystemExit:
                raise
            except Exception as e:
                print(f"[TEST] {label} -> exception: {e}")

print("\n[FAILED] 没有找到 OpenCV 可打开的摄像头")
