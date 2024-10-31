import argparse
import json
import os
import sys
from multiprocessing import Process, Queue
import multiprocessing

import cv2
import numpy as np
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

sys.path.append("..")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)
CORS(app)

multiprocessing.set_start_method('spawn', force=True)


def sam_worker(q: Queue, ack: Queue, args: argparse.Namespace) -> None:
    """
    Worker process for SAM model predictions.
    
    Args:
        q: Queue for receiving annotation requests
        ack: Queue for sending acknowledgments
        args: Command line arguments containing paths
    """
    scenes_directory = args.scenes_directory
    checkpoint_path = args.checkpoint_path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path))

    while True:
        data = q.get()
        
        # Extract data from request
        bbox = data["bbox"]
        positive_keypoints = data["positiveKeypoints"]
        negative_keypoints = data["negativeKeypoints"]
        scene_name = data["sceneName"]
        
        # Process bounding box coordinates
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        # Load and process image
        image_path = os.path.join(scenes_directory, scene_name, "image.png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image)

        # Prepare input data
        input_box = np.array([[x1, y1, x2, y2]])
        input_point = []
        input_label = []

        for point in positive_keypoints:
            input_point.append(point)
            input_label.append(1)

        for point in negative_keypoints:
            input_point.append(point)
            input_label.append(0)

        # Generate prediction
        masks, _, _ = predictor.predict(
            point_coords=np.array(input_point) if input_point else None,
            point_labels=np.array(input_label) if input_label else None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Save results
        mask = masks[0]
        mask_image_path = os.path.join(scenes_directory, scene_name, "mask.png")
        cv2.imwrite(mask_image_path, ((mask > 0) * 255).astype(np.uint8))

        result = {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "positiveKeypoints": positive_keypoints,
            "negativeKeypoints": negative_keypoints,
        }
        
        result_path = os.path.join(scenes_directory, scene_name, "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f)

        print(f"Processed scene: {scene_name}")
        ack.put(scene_name)


def create_app(scenes_directory: str, q: Queue, ack: Queue, queued_scenes: set) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        scenes_directory: Path to the directory containing scenes
        q: Queue for sending requests to worker
        ack: Queue for receiving acknowledgments
        queued_scenes: Set to track queued scene requests
    
    Returns:
        Flask application instance
    """
    @app.route("/scenes/<scene_name>/image")
    def get_image(scene_name: str):
        image_path = os.path.join(scenes_directory, scene_name, "image.png")
        return send_file(image_path, mimetype="image/png")

    @app.route("/scenes/<scene_name>/mask")
    def get_mask(scene_name: str):
        try:
            mask_path = os.path.join(scenes_directory, scene_name, "mask.png")
            return send_file(mask_path, mimetype="image/png")
        except FileNotFoundError:
            return jsonify({"error": "Mask not found"}), 404

    @app.route("/api/scenes", methods=["GET"])
    def get_scenes():
        scene_names = [
            name for name in sorted(os.listdir(scenes_directory))
            if os.path.isdir(os.path.join(scenes_directory, name))
        ]

        scenes_info = []
        for scene_name in scene_names:
            result_path = os.path.join(scenes_directory, scene_name, "result.json")
            result = None
            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
            
            scenes_info.append({
                "sceneName": scene_name,
                "result": result,
            })

        return jsonify({"scenes": scenes_info})

    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        q.put(data)

        while True:
            processed_scene = ack.get()
            if processed_scene == data["sceneName"]:
                return jsonify({})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server for scene annotation")
    parser.add_argument("scenes_directory", type=str, help="Path to the directory containing the scenes")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    q = Queue()
    ack = Queue()
    queued_scenes = set()

    worker = Process(target=sam_worker, args=(q, ack, args))
    worker.start()

    app = create_app(args.scenes_directory, q, ack, queued_scenes)
    app.run(debug=True, host="localhost", use_reloader=False)

    worker.join()