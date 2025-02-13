import argparse
import os
import shutil
import yaml
import glob
import json
import logging
from tqdm import tqdm
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define class mapping
class_mapping = {
    "bus": 0, "traffic light": 1, "traffic sign": 2, "person": 3, "bike": 4,
    "truck": 5, "motor": 6, "car": 7, "train": 8, "rider": 9
}

def convert_annotations_to_yolo(json_dir, img_dir, output_dir):
    """
    Convert annotations from BDD format to YOLO format.

    Args:
        json_dir (str): Path to the directory containing JSON annotation files.
        img_dir (str): Path to the directory containing images.
        output_dir (str): Directory to save YOLO formatted annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    logging.info(f"Found {len(json_files)} annotation files in {json_dir}.")
    
    for json_file in tqdm(json_files, desc="Converting annotations"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for frame in data["frames"]:
            img_filename = os.path.splitext(os.path.basename(json_file))[0] + ".jpg"
            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                logging.warning(f"Image file {img_filename} not found. Skipping.")
                continue
            shutil.copy(img_path, os.path.join(images_output_dir, img_filename))
            
            label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")
            with open(label_file, "w") as lf:
                for obj in frame["objects"]:
                    if "box2d" in obj:
                        category = obj["category"]
                        if category not in class_mapping:
                            continue
                        class_id = class_mapping[category]
                        x1, y1, x2, y2 = obj["box2d"].values()
                        img_w, img_h = 1280, 720
                        x_center, y_center = (x1 + x2) / (2 * img_w), (y1 + y2) / (2 * img_h)
                        w_norm, h_norm = (x2 - x1) / img_w, (y2 - y1) / img_h
                        lf.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
    logging.info(f"Converted annotations saved in {output_dir}.")

def main():
    """
    Main function to parse arguments and run the YOLO training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO model on BDD dataset")
    parser.add_argument("--data_dir", type=str, default="100k", help="Path to dataset directory")
    parser.add_argument("--yolo_data_dir", type=str, default="yolo_dataset", help="Output directory for YOLO formatted data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
    parser.add_argument("--output_dir", type=str, default="result", help="Output directory for training results")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of data to be used for training")
    args = parser.parse_args()
    
    logging.info("Preparing dataset...")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    
    data_splits = {"train": train_dir, "val": val_dir, "test": test_dir}
    for split, folder in data_splits.items():
        convert_annotations_to_yolo(folder, folder, os.path.join(args.yolo_data_dir, split))
    
    data_yaml = {
        "train": os.path.abspath(os.path.join(args.yolo_data_dir, "train/images")),
        "val": os.path.abspath(os.path.join(args.yolo_data_dir, "val/images")),
        "nc": len(class_mapping),
        "names": list(class_mapping.keys())
    }
    yaml_path = os.path.join(args.yolo_data_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    logging.info(f"YOLO dataset configuration saved at {yaml_path}.")
    
    logging.info("Starting model training...")
    model = YOLO("yolov8m.pt")
    model.train(data=yaml_path, epochs=args.epochs, imgsz=args.img_size, batch=args.batch_size,
                save=True, project=args.output_dir, save_period=3, fraction=args.fraction)
    logging.info("Model training completed.")

if __name__ == "__main__":
    main()
