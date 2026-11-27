"""Recover the 6-point 3D face model that MPIIFaceGaze used for its head-pose labels.

The download has no face-model file, but every annotation line holds the 2D landmarks
and the head pose (R, t) that were fitted with that model. For each landmark k:

    image point  ~  K (R X_k + t)

which is linear in the unknown 3D point X_k. Stacking thousands of images gives a
least-squares estimate of each X_k. The live tracker uses the result with solvePnP,
so live head pose matches the training labels.

Check: the readme says fc = mean of the six landmarks in camera coordinates,
so mean(R X + t) should reproduce fc (printed as the fc error, in mm).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

# Landmark order in pXX.txt, as seen in the image from left to right:
# eye corner, eye corner, eye corner, eye corner, mouth corner, mouth corner.


def load_subject(subject_dir: Path):
    sid = subject_dir.name
    mat = loadmat(str(subject_dir / "Calibration" / "Camera.mat"))
    K = mat["cameraMatrix"].astype(np.float64)
    dist = mat["distCoeffs"].astype(np.float64)

    lm, R, t, fc = [], [], [], []
    for line in (subject_dir / f"{sid}.txt").read_text().splitlines():
        p = line.split()
        lm.append(np.array(p[3:15], dtype=np.float64).reshape(6, 2))
        R.append(cv2.Rodrigues(np.array(p[15:18], dtype=np.float64))[0])
        t.append(np.array(p[18:21], dtype=np.float64))
        fc.append(np.array(p[21:24], dtype=np.float64))
    lm = np.stack(lm)

    # Undistort and convert pixels to normalized camera coordinates (a, b).
    norm = cv2.undistortPoints(lm.reshape(-1, 1, 2), K, dist).reshape(lm.shape)
    return norm, np.stack(R), np.stack(t), np.stack(fc)


def solve_points(norm, R, t):
    """Least-squares X_k for each of the 6 landmarks.

    a (r3.X + t3) = r1.X + t1   ->   (r1 - a r3).X = a t3 - t1
    b (r3.X + t3) = r2.X + t2   ->   (r2 - b r3).X = b t3 - t2
    """
    X = np.zeros((6, 3))
    for k in range(6):
        a, b = norm[:, k, 0], norm[:, k, 1]
        A = np.concatenate([R[:, 0, :] - a[:, None] * R[:, 2, :],
                            R[:, 1, :] - b[:, None] * R[:, 2, :]])
        y = np.concatenate([a * t[:, 2] - t[:, 0],
                            b * t[:, 2] - t[:, 1]])
        X[k] = np.linalg.lstsq(A, y, rcond=None)[0]
    return X


def fc_error(X, R, t, fc):
    pred = (R @ X.T).mean(axis=2) + t          # mean of R X_k + t over the 6 points
    return np.linalg.norm(pred - fc, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Data")
    ap.add_argument("--out", default="face_model.npy")
    args = ap.parse_args()

    all_norm, all_R, all_t, all_fc = [], [], [], []
    for i in range(15):
        sid = f"p{i:02d}"
        norm, R, t, fc = load_subject(Path(args.src) / sid)
        X = solve_points(norm, R, t)
        print(f"{sid}: fc error median {np.median(fc_error(X, R, t, fc)):.2f} mm")
        all_norm.append(norm); all_R.append(R); all_t.append(t); all_fc.append(fc)

    norm, R, t, fc = map(np.concatenate, (all_norm, all_R, all_t, all_fc))
    X = solve_points(norm, R, t)
    err = fc_error(X, R, t, fc)
    print(f"ALL: fc error median {np.median(err):.2f} mm, 95th percentile {np.percentile(err, 95):.2f} mm")
    print("3D face model (mm), rows = landmarks in pXX.txt order:")
    print(np.round(X, 2))
    np.save(args.out, X)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
