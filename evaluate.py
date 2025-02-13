import argparse
import os
import logging
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml):
    """
    Evaluate a trained YOLO model on the validation dataset.

    Args:
        model_path (str): Path to the trained YOLO model weights.
        data_yaml (str): Path to the dataset YAML configuration file.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please check the path.")
        exit(1)
    
    logging.info("Loading YOLO model...")
    model = YOLO(model_path)
    
    try:
        logging.info("Starting model evaluation on validation dataset...")
        results = model.val(data=data_yaml)
        
        # Log evaluation metrics
        logging.info(f"Class indices with average precision: {results.ap_class_index}")
        logging.info(f"Average precision for all classes: {results.box.all_ap}")
        logging.info(f"Overall average precision: {results.box.ap}")
        logging.info(f"Mean Average Precision (mAP) @ IoU=0.50: {results.box.map50}")
        logging.info(f"Mean Recall (MR): {results.box.mr}")
    
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--model_path", type=str, default="results/train11/weights/best.pt", 
                        help="Path to the trained YOLO model weights")
    parser.add_argument("--data_yaml", type=str, default="yolo_dataset/data.yaml", 
                        help="Path to the dataset YAML configuration file")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_yaml)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
