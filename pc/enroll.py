

import argparse
import json
import os
import time

import cv2
import numpy as np

from face_capture import capture_face_vector, NoFaceDetected
from pca_eigenfaces import EigenfaceModel
from serial_protocol import VECTOR_LEN


def load_identities(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_identities(path, identities):
    with open(path, "w") as f:
        json.dump(identities, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Nome da identidade (max 11 caracteres ASCII)")
    ap.add_argument("--model", default="model.npz")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="identities.json")
    ap.add_argument("--delay", type=float, default=0.6,
                     help="segundos entre capturas, pra variar levemente pose/expressão")
    args = ap.parse_args()

    if len(args.name.encode("ascii")) > 11:
        raise SystemExit("Nome precisa ter no máximo 11 caracteres ASCII (limite da tabela no Arduino).")

    model = EigenfaceModel.load(args.model)
    if model.n_components != VECTOR_LEN:
        raise SystemExit(
            f"Modelo tem {model.n_components} componentes, mas o protocolo serial "
            f"espera {VECTOR_LEN}. Retreine com --components {VECTOR_LEN} em train_pca.py."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}.")

    print(f"Cadastrando '{args.name}' — capturando {args.samples} fotos. "
          "Varie levemente a pose/expressão entre uma captura e outra.")

    vectors = []
    attempts = 0
    while len(vectors) < args.samples and attempts < args.samples * 10:
        attempts += 1
        try:
            vec = capture_face_vector(cap, show_preview=True)
            vectors.append(vec)
            print(f"  [{len(vectors)}/{args.samples}] captura ok")
            time.sleep(args.delay)
        except NoFaceDetected:
            continue

    cap.release()
    cv2.destroyAllWindows()

    if len(vectors) < args.samples:
        raise SystemExit(
            f"Só consegui {len(vectors)}/{args.samples} capturas com rosto detectado. "
            "Verifique iluminação/enquadramento e rode de novo."
        )

    X = np.array(vectors)
    quantized = model.project_quantized(X)
    reference = np.round(quantized.mean(axis=0)).astype(int).tolist()

    dists = [np.linalg.norm(q.astype(np.int32) - np.array(reference)) for q in quantized]
    print(f"Dispersão intra-amostra (enrollment): média={np.mean(dists):.1f}, "
          f"máx={np.max(dists):.1f}")

    identities = load_identities(args.out)
    identities[args.name] = reference
    save_identities(args.out, identities)

    print(f"Identidade '{args.name}' salva em {args.out} "
          f"({len(identities)} identidade(s) cadastrada(s) no total).")


if __name__ == "__main__":
    main()
