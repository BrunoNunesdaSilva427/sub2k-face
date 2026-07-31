
import argparse
import glob
import os
import sys

import cv2
import numpy as np

from face_capture import detect_and_normalize_face, NoFaceDetected
from pca_eigenfaces import EigenfaceModel


def load_training_vectors(images_dir):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not paths:
        sys.exit(f"Nenhuma imagem encontrada em {images_dir}")

    vectors = []
    skipped = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue
        try:
            vec, _ = detect_and_normalize_face(img)
            vectors.append(vec)
        except NoFaceDetected:
            skipped += 1

    print(f"{len(vectors)} rostos aproveitados, {skipped} imagens descartadas "
          f"(sem rosto detectado ou ilegíveis).")

    if len(vectors) < 20:
        print("AVISO: menos de 20 amostras — a base PCA pode ficar instável/"
              "enviesada. Recomendo pelo menos algumas dezenas de fotos variadas.")

    return np.array(vectors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", default="model.npz")
    ap.add_argument("--components", type=int, default=32,
                     help="precisa bater com VECTOR_LEN em serial_protocol.py (32)")
    args = ap.parse_args()

    X = load_training_vectors(args.images_dir)

    model = EigenfaceModel(n_components=args.components).fit(X)
    model.save(args.out)

    print(f"Modelo treinado e salvo em {args.out} "
          f"({args.components} componentes, {X.shape[0]} amostras de treino).")


if __name__ == "__main__":
    main()
