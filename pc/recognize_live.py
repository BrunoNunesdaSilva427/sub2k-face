
import argparse
import time

import cv2

from face_capture import capture_face_vector, NoFaceDetected
from pca_eigenfaces import EigenfaceModel
from arduino_link import ArduinoLink
from serial_protocol import VECTOR_LEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.npz")
    ap.add_argument("--port", required=True, help="ex: /dev/ttyUSB0, /dev/ttyACM0, COM3")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5,
                     help="segundos entre tentativas de reconhecimento")
    args = ap.parse_args()

    model = EigenfaceModel.load(args.model)
    if model.n_components != VECTOR_LEN:
        raise SystemExit(
            f"Modelo tem {model.n_components} componentes, mas o protocolo serial "
            f"espera {VECTOR_LEN}. Retreine com --components {VECTOR_LEN} em train_pca.py."
        )

    print(f"Conectando ao Arduino em {args.port}...")
    link = ArduinoLink(args.port, baudrate=args.baudrate)
    print("Conectado.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}.")

    print("Reconhecimento ao vivo iniciado. Ctrl+C pra sair.")
    try:
        while True:
            try:
                vec = capture_face_vector(cap, show_preview=True)
            except NoFaceDetected:
                time.sleep(args.interval)
                continue

            qvec = model.project_quantized(vec[None, :])[0]

            try:
                matched, name = link.query(qvec)
            except TimeoutError as e:
                print(f"[erro serial] {e}")
                time.sleep(args.interval)
                continue

            if matched:
                print(f"✔ Reconhecido: {name}")
            else:
                print("✘ Desconhecido")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        link.close()


if __name__ == "__main__":
    main()
