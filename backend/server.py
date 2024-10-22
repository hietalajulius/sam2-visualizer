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
        print("Waiting for data")
        data = q.get()
        print("Data received", data)

        manual = data["manual"]

        bbox = data["bbox"]
        positive_keypoints = data["positiveKeypoints"]
        negative_keypoints = data["negativeKeypoints"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        if "outlierThreshold" in data and data["outlierThreshold"] is not None:
            outlier_threshold = float(data["outlierThreshold"])
        else:
            outlier_threshold = 0.5

        scene_name = data["sceneName"]

        image = cv2.imread(f"{scenes_directory}/{scene_name}/observation_result/image_left.png")
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

        depth_map_path = f"{scenes_directory}/{scene_name}/observation_result/depth_map.tiff"
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

        masked_depth_map = np.where(mask > 0, depth_map, 0)
        masked_values = masked_depth_map[mask > 0]

        mean = np.mean(masked_values)

        lower_bound = mean - outlier_threshold
        upper_bound = mean + outlier_threshold

        masked_depth_map = np.where(
            (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
        )

        x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))

        intrinsics = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json"))

        cx = intrinsics["principal_point_in_pixels"]["cx"]
        cy = intrinsics["principal_point_in_pixels"]["cy"]

        fx = intrinsics["focal_lengths_in_pixels"]["fx"]
        fy = intrinsics["focal_lengths_in_pixels"]["fy"]

        X1 = (x - 0.5 - cx) * masked_depth_map / fx
        Y1 = (y - 0.5 - cy) * masked_depth_map / fy
        X2 = (x + 0.5 - cx) * masked_depth_map / fx
        Y2 = (y + 0.5 - cy) * masked_depth_map / fy

        pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))

        mask_image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"

        cv2.imwrite(mask_image_path, ((pixel_areas > 0) * 255).astype(np.uint8))

        coverage = np.nan_to_num(np.sum(pixel_areas), nan=0, posinf=0, neginf=0)

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
                "outlierThreshold": outlier_threshold,
                "coverage": coverage,
            },
            open(f"{scenes_directory}/{scene_name}/observation_result/result.json", "w"),
        )

        print("Data processed")
        if manual:
            ack.put(scene_name)


def create_app(scenes_directory, q, ack, queued_scenes):  # noqa C901
    @app.route("/")
    def index():
        root_dir = "./static"  # Directory you want to explore
        if os.path.isdir(root_dir):
            files = sorted(os.listdir(root_dir))
            return render_template("explorer.html", root=root_dir, files=files)

    @app.route("/explore")
    def explore():
        root_dir = request.args.get("dir", ".")
        if os.path.isdir(root_dir):
            files = sorted(os.listdir(root_dir))
            return render_template("explorer.html", root=root_dir, files=files)
        else:
            return "Invalid directory"

    ALLOWED_EXTENSIONS = {"json"}

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/image")
    def get_image(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{scenes_directory}/{scene_name}/observation_result/image_left.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type

    @app.route("/scenes/<scene_name>/depth")
    def get_depth(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{scenes_directory}/{scene_name}/observation_result/depth_image.jpg"
        return send_file(image_path, mimetype="image/jpeg")  # Adjust mimetype as per your image type

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/mask")
    def get_mask(scene_name):
        # Construct the image path using the provided directory
        try:
            image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"
            return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type
        except FileNotFoundError:
            return jsonify({"error": "Mask not found"})

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/coverage")
    def get_coverage(scene_name):
        # Construct the image path using the provided directory

        coverage = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/coverage.json"))

        return jsonify(coverage)

    @app.route("/api/scenes", methods=["GET"])
    def get_scenes():
        scene_names = sorted(os.listdir(scenes_directory))
        scene_names = [
            scene_name
            for scene_name in scene_names
            if os.path.isdir(f"{scenes_directory}/{scene_name}") and scene_name.startswith("sample_")
        ]

        logger.info(f"Found {len(scene_names)} in {os.path.abspath(scenes_directory)}")

        scenes_info = []

        for scene_name in scene_names:
            dataset_path = os.path.join(scenes_directory, scene_name)
            last_modified_time = get_last_modified_time(dataset_path)

            result_exists = os.path.exists(f"{scenes_directory}/{scene_name}/observation_result/result.json")
            result = None

            if result_exists:
                result = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/result.json"))
            elif scene_name not in queued_scenes:
                intrinsics = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json")
                )
                extrinsics = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/camera_pose_in_world.json")
                )
                tcp_left = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/arm_left_tcp_pose_in_world.json")
                )
                tcp_right = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/arm_right_tcp_pose_in_world.json")
                )

                cx = intrinsics["principal_point_in_pixels"]["cx"]
                cy = intrinsics["principal_point_in_pixels"]["cy"]

                fx = intrinsics["focal_lengths_in_pixels"]["fx"]
                fy = intrinsics["focal_lengths_in_pixels"]["fy"]

                position = extrinsics["position_in_meters"]
                rotation = extrinsics["rotation_euler_xyz_in_radians"]

                # Convert rotation from Euler angles to rotation matrix
                R = euler_to_mat(rotation["roll"], rotation["pitch"], rotation["yaw"])

                # Convert position to numpy array
                T = np.array([position["x"], position["y"], position["z"]])

                tcp_left_position = tcp_left["position_in_meters"]
                tcp_right_position = tcp_right["position_in_meters"]

                x_left = tcp_left_position["x"]
                y_left = tcp_left_position["y"]
                z_left = tcp_left_position["z"]

                x_right = tcp_right_position["x"]
                y_right = tcp_right_position["y"]
                z_right = tcp_right_position["z"]

                # Create the 3D rectangle for the bounding box
                y_padding = 0.1
                c1 = np.array([x_left, y_left + y_padding, z_left])
                c2 = np.array([x_right, y_right - y_padding, z_right])
                c3 = np.array([x_left, y_left + y_padding, 0.05])
                c4 = np.array([x_right, y_right - y_padding, 0.05])

                # Generate all corners
                corners = [c1, c2, c3, c4]

                projected_corners = []
                for corner in corners:
                    # Convert world coordinates to camera coordinates
                    X_cam = R.T @ (np.array(corner) - T)

                    # Project to the image plane
                    u = fx * (X_cam[0] / X_cam[2]) + cx
                    v = fy * (X_cam[1] / X_cam[2]) + cy

                    print("Projected corner", u, v, "for corner", corner)

                    projected_corners.append((u, v))

                # Get the 2D bounding box
                u_min = min(u for u, _ in projected_corners)
                v_min = min(v for _, v in projected_corners)
                u_max = max(u for u, _ in projected_corners)
                v_max = max(v for _, v in projected_corners)

                print("Queueing unannotated scene", scene_name, "with bounding box", u_min, v_min, u_max, v_max)

                q.put(
                    {
                        "sceneName": scene_name,
                        "bbox": {"x1": u_min, "y1": v_min, "x2": u_max, "y2": v_max},
                        "positiveKeypoints": [],
                        "negativeKeypoints": [],
                        "outlierThreshold": 0.5,
                        "manual": False,
                    }
                )
                queued_scenes.add(scene_name)

            scenes_info.append(
                {
                    "sceneName": scene_name,
                    "lastModifiedTime": last_modified_time,
                    "result": result,
                }
            )

        return jsonify({"scenes": scenes_info})

    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        data["manual"] = True
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
