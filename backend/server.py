import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from loguru import logger
from werkzeug.utils import secure_filename

sys.path.append("..")
import datetime
import glob
import json
import os

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)
CORS(app)  #

from multiprocessing import Process, Queue
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)


def sam_worker(q, ack, args):
    scenes_directory = args.scenes_directory

    checkpoint = "/home/julius/dev/sam2-visualizer/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    while True:
        data = q.get()

        bbox = data["bbox"]
        positive_keypoints = data["positiveKeypoints"]
        negative_keypoints = data["negativeKeypoints"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        scene_name = data["sceneName"]

        image = cv2.imread(f"{scenes_directory}/{scene_name}/image.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        input_box = np.array([[x1, y1, x2, y2]])

        input_point = []
        input_label = []

        for point in positive_keypoints:
            input_point.append(point)
            input_label.append(1)

        for point in negative_keypoints:
            input_point.append(point)
            input_label.append(0)

        logger.info(
            f"Predicting for scene {scene_name} with box {input_box} and input points {input_point} and labels {input_label}"
        )

        masks, _, _ = predictor.predict(
            point_coords=np.array(input_point) if len(input_point) > 0 else None,
            point_labels=np.array(input_label) if len(input_label) > 0 else None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0]

        mask_image_path = f"{scenes_directory}/{scene_name}/mask.png"

        cv2.imwrite(mask_image_path, ((mask > 0) * 255).astype(np.uint8))

        json.dump(
            {
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "positiveKeypoints": positive_keypoints,
                "negativeKeypoints": negative_keypoints,
            },
            open(f"{scenes_directory}/{scene_name}/result.json", "w"),
        )

        print("Data processed")
        ack.put(scene_name)


def create_app(scenes_directory, q, ack, queued_scenes):  # noqa C901
    # @app.route("/")
    # def index():
    #     root_dir = "./static"  # Directory you want to explore
    #     if os.path.isdir(root_dir):
    #         files = sorted(os.listdir(root_dir))
    #         return render_template("explorer.html", root=root_dir, files=files)

    # @app.route("/explore")
    # def explore():
    #     root_dir = request.args.get("dir", ".")
    #     if os.path.isdir(root_dir):
    #         files = sorted(os.listdir(root_dir))
    #         return render_template("explorer.html", root=root_dir, files=files)
    #     else:
    #         return "Invalid directory"

    # ALLOWED_EXTENSIONS = {"json"}

    # def allowed_file(filename):
    #     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/image")
    def get_image(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{scenes_directory}/{scene_name}/image.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/mask")
    def get_mask(scene_name):
        # Construct the image path using the provided directory
        try:
            image_path = f"{scenes_directory}/{scene_name}/mask.png"
            return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type
        except FileNotFoundError:
            return jsonify({"error": "Mask not found"})

    @app.route("/api/scenes", methods=["GET"])
    def get_scenes():
        scene_names = sorted(os.listdir(scenes_directory))
        scene_names = [
            scene_name
            for scene_name in scene_names
            if os.path.isdir(f"{scenes_directory}/{scene_name}")
        ]

        logger.info(f"Found {len(scene_names)} in {os.path.abspath(scenes_directory)}")

        scenes_info = []

        for scene_name in scene_names:

            result_exists = os.path.exists(f"{scenes_directory}/{scene_name}/result.json")
            result = None

            if result_exists:
                result = json.load(open(f"{scenes_directory}/{scene_name}/result.json"))

            scenes_info.append(
                {
                    "sceneName": scene_name,
                    "result": result,
                }
            )

        return jsonify({"scenes": scenes_info})

    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        q.put(data)

        logger.info(f"Manually queueing scene {data['sceneName']} for annotation")

        while True:
            logger.info(f"Waiting for scene {data['sceneName']} to be processed")
            processed_scene = ack.get()
            logger.info(f"Processed scene {processed_scene} asked for {data['sceneName']}")

            if processed_scene == data["sceneName"]:
                logger.info(f"Scene {data['sceneName']} processed")
                return jsonify({})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server to serve the competition scenes")
    parser.add_argument("scenes_directory", type=str, help="Path to the directory containing the scenes")

    args = parser.parse_args()

    q = Queue()
    ack = Queue()

    p = Process(
        target=sam_worker,
        args=(
            q,
            ack,
            args,
        ),
    )
    p.start()

    queued_scenes = set()

    app = create_app(args.scenes_directory, q, ack, queued_scenes)
    app.run(debug=True, host="localhost", use_reloader=False)
    # app.run(debug=True, host="10.42.0.1", use_reloader=False)

    p.join()
