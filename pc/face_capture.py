

import cv2
import numpy as np

IMG_SIZE = 32  

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class NoFaceDetected(Exception):
    pass


def detect_and_normalize_face(frame_bgr, margin=0.2):
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )

    if len(faces) == 0:
        raise NoFaceDetected("Nenhum rosto detectado no frame.")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(gray_full.shape[1], x + w + mx), min(gray_full.shape[0], y + h + my)

    face_crop = gray_full[y0:y1, x0:x1]
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    face_eq = cv2.equalizeHist(face_resized)

    return face_eq.astype(np.float64).flatten() / 255.0, (x0, y0, x1, y1)


def capture_face_vector(cap, show_preview=False):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Falha ao ler frame da câmera.")

    vector, bbox = detect_and_normalize_face(frame)

    if show_preview:
        x0, y0, x1, y1 = bbox
        preview = frame.copy()
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("captura", preview)
        cv2.waitKey(1)

    return vector
