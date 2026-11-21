"""Normalize MPIIFaceGaze into 224x224 face crops with (pitch, yaw) labels.

Method: Zhang, Sugano, Bulling, "Revisiting Data Normalization for
Appearance-Based Gaze Estimation", ETRA 2018.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

FOCAL_N = 960     # virtual camera focal length (pixels)
DIST_N = 600.0    # virtual camera distance to the face center (mm)
SIZE = 224        # output crop size (pixels)

CAM_N = np.array([[FOCAL_N, 0, SIZE / 2],
                  [0, FOCAL_N, SIZE / 2],
                  [0, 0, 1]], dtype=np.float64)

def load_camera(subject_dir: Path) -> np.ndarray:
    """3x3 intrinsic matrix of this participant's laptop webcam."""
    mat = loadmat(str(subject_dir / "Calibration" / "Camera.mat"))
    return mat["cameraMatrix"].astype(np.float64)


def parse_line(line: str):
    """One annotation line -> (relative image path, hr, fc, gt)."""
    p = line.split()
    rel_path = p[0]
    hr = np.array(p[15:18], dtype=np.float64)   # head rotation -> Rodrigues vector
    fc = np.array(p[21:24], dtype=np.float64)   # face center, camera coords (mm)
    gt = np.array(p[24:27], dtype=np.float64)   # gaze target, camera coords (mm)
    return rel_path, hr, fc, gt


def normalization_rotation(hr: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Rotation R (3x3) from the real camera frame to the normalized camera frame."""
    R_head = cv2.Rodrigues(hr)[0]
    head_x = R_head[:, 0]
    z = fc / np.linalg.norm(fc)
    y = np.cross(z, head_x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    return np.stack([x, y, z])


def gaze_to_pitchyaw(g: np.ndarray):
    """Unit gaze vector in the normalized frame -> (pitch, yaw) in radians."""
    pitch = np.arcsin(-g[1])
    yaw = np.arctan2(-g[0], -g[2])
    return float(pitch), float(yaw)


def normalize(img: np.ndarray, hr, fc, gt, cam: np.ndarray):
    """Returns (224x224 BGR face crop, pitch, yaw)."""
    R = normalization_rotation(hr, fc)
    # Step 6: scale so the face sits at DIST_N mm from the virtual camera.
    S = np.diag([1.0, 1.0, DIST_N / np.linalg.norm(fc)])

    # Step 7: real image pixels -> real camera rays -> rotate + scale -> virtual camera pixels.
    W = CAM_N @ S @ R @ np.linalg.inv(cam)

    # Step 8: warp the full frame into a 224x224 normalized face crop.
    face = cv2.warpPerspective(img, W, (SIZE, SIZE))

    # Step 9: gaze direction rotates with the camera; scaling does not change direction.
    g = R @ (gt - fc)
    g /= np.linalg.norm(g)

    pitch, yaw = gaze_to_pitchyaw(g)
    return face, pitch, yaw


def process_subject(subject_dir: Path, out_dir: Path, writer, limit=None):
    sid = subject_dir.name
    cam = load_camera(subject_dir)
    (out_dir / sid).mkdir(parents=True, exist_ok=True)

    lines = (subject_dir / f"{sid}.txt").read_text().splitlines()
    if limit:
        lines = lines[:limit]

    for line in lines:
        rel, hr, fc, gt = parse_line(line)
        img = cv2.imread(str(subject_dir / rel))
        if img is None:
            raise FileNotFoundError(subject_dir / rel)

        face, pitch, yaw = normalize(img, hr, fc, gt, cam)

        out_rel = f"{sid}/{rel.replace('/', '_')}"    
        cv2.imwrite(str(out_dir / out_rel), face)
        writer.writerow([sid, out_rel, f"{pitch:.6f}", f"{yaw:.6f}"])

    print(f"{sid}: {len(lines)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Data")
    ap.add_argument("--out", default="data/normalized")
    ap.add_argument("--subjects", nargs="*", default=[f"p{i:02d}" for i in range(15)])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images per subject (for testing)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "path", "pitch", "yaw"])
        for sid in args.subjects:
            process_subject(src / sid, out, writer, args.limit)
            
if __name__ == "__main__":
    main()
