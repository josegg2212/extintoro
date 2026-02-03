from ultralytics import YOLO
import os
import sys


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.environ.get(
        "YOLO_WEIGHTS",
        os.path.join(base_dir, "modelo", "train_fuego", "weights", "best.pt"),
    )
    test_images_dir = os.environ.get(
        "YOLO_TEST_DIR",
        os.path.join(base_dir, "dataset", "test", "images"),
    )
    project_dir = os.environ.get("YOLO_PREDICT_DIR", os.path.join(base_dir, "temp"))
    run_name = os.environ.get("YOLO_PREDICT_NAME", "predict")

    if not os.path.exists(weights_path):
        print(f"No encuentro pesos en: {weights_path}")
        return 1
    if not os.path.exists(test_images_dir):
        print(f"No encuentro imágenes de test en: {test_images_dir}")
        return 1

    model = YOLO(weights_path)
    model.predict(
        source=test_images_dir,
        save=True,
        project=project_dir,
        name=run_name,
    )
    print(f"Predicciones guardadas en: {os.path.join(project_dir, run_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
