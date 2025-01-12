// App.js
import "./App.css";
import React, { useState, useEffect, useRef } from "react";
import { Stage, Layer, Image, Rect, Circle } from "react-konva";
import axios from "axios";
import Select from "react-select";
import Switch from "react-switch";
import ClipLoader from "react-spinners/ClipLoader";
import Modal from "react-modal";
import _ from "lodash";

const customStyles = {
  option: (provided, state) => ({
    ...provided,
    color: "black",
    backgroundColor: state.isSelected ? "lightGrey" : "white",
  }),
  control: (provided) => ({
    ...provided,
    backgroundColor: "white",
    color: "black",
  }),
};

var colorReplaceFilter = function (imageData) {
  var nPixels = imageData.data.length;
  for (var i = 0; i < nPixels; i += 4) {
    const isWhite =
      imageData.data[i] === 255 &&
      imageData.data[i + 1] === 255 &&
      imageData.data[i + 2] === 255;
    if (isWhite) {
      imageData.data[i] = 0; // Red
      imageData.data[i + 1] = 255; // Green
      imageData.data[i + 2] = 0; // Blue
    }
  }
};

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [annotationIsLoading, setAnnotationIsLoading] = useState(false);

  const [showMask, setShowMask] = useState(true);

  const [currentScene, setCurrentScene] = useState(null);
  const [image, setImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);

  const [availableScenes, setAvailableScenes] = useState([]);

  const [rect, setRect] = useState(null);
  const [drawing, setDrawing] = useState(false);

  const [positiveKeypoints, setPositiveKeypoints] = useState([]);
  const [negativeKeypoints, setNegativeKeypoints] = useState([]);

  const [annotationMode, setAnnotationMode] = useState("box");

  const maskImageRef = React.useRef();

  const scale = image?.scale;

  const setAnnotationsFromResult = (result, image) => {
    if (!result || !image?.scale) return;

    setRect({
      x: result.bbox.x1 * scale,
      y: result.bbox.y1 * scale,
      width: (result.bbox.x2 - result.bbox.x1) * scale,
      height: (result.bbox.y2 - result.bbox.y1) * scale,
    });

    setPositiveKeypoints(
      result.positiveKeypoints.map((pos) => ({
        x: pos[0] * scale,
        y: pos[1] * scale,
      }))
    );

    setNegativeKeypoints(
      result.negativeKeypoints.map((pos) => ({
        x: pos[0] * scale,
        y: pos[1] * scale,
      }))
    );
  };

  useEffect(() => {
    const result = currentScene?.result;
    setAnnotationsFromResult(result, image);
  }, [image, currentScene]);

  useEffect(() => {
    updateAvailableScenes();
  }, []);

  React.useEffect(() => {
    if (maskImage) {
      maskImageRef.current?.cache();
    }
  }, [maskImage, showMask]);

  const updateImageSources = (sceneName) => {
    const maxWidth = window.innerWidth;

    try {
      const maskImg = new window.Image();
      maskImg.src = `http://127.0.0.1:5000/scenes/${sceneName}/mask?${Date.now()}`;
      maskImg.crossOrigin = "Anonymous";
      maskImg.onload = () => {
        const scale = maxWidth / maskImg.width;
        setMaskImage({ img: maskImg, scale });
      };
    } catch (error) {
      console.error("Error fetching mask image:", error);
    }

    const img = new window.Image();
    img.src = `http://127.0.0.1:5000/scenes/${sceneName}/image`;
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const scale = maxWidth / img.width;
      setImage({ img, scale });
    };
  };

  const updateAvailableScenes = async () => {
    try {
      const loadAvailableScenes = await getAvailableScenes();
      console.log("start load");
      setAvailableScenes(loadAvailableScenes);
    } catch (error) {
      console.error("Error fetching available files:", error);
    } finally {
      if (isLoading) {
        setIsLoading(false);
      }
    }
  };

  const handleRefresh = async () => {
    updateAvailableScenes();
    setAnnotationsFromResult(currentScene?.result, image);
  };

  const handleClearManualAnnotations = async () => {
    setPositiveKeypoints([]);
    setNegativeKeypoints([]);
    setRect(null);
  };

  useEffect(() => {
    const sortedScenes = availableScenes.sort(
      (a, b) => a.sceneName - b.sceneName
    );

    if (sortedScenes.length > 0) {
      const sceneToSet = sortedScenes[0];
      if (!currentScene) {
        setCurrentScene(sceneToSet);
        setMaskImage(null);
        updateImageSources(sceneToSet.sceneName);
        return;
      }

      const updatedScene = availableScenes.find(
        (scene) => scene.sceneName === currentScene.sceneName
      );

      if (!_.isEqual(updatedScene, currentScene)) {
        setCurrentScene(updatedScene);
        setMaskImage(null);
        updateImageSources(updatedScene.sceneName);
      }
    }
  }, [availableScenes, currentScene]);

  const getAvailableScenes = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/scenes");
      return response?.data?.scenes ?? [];
    } catch (error) {
      console.error("Error fetching available files:", error);
      return [];
    }
  };

  const handleChange = (selectedOption) => {
    setCurrentScene(selectedOption.value);
    handleClearManualAnnotations();
    setMaskImage(null);
    updateImageSources(selectedOption.value.sceneName);
  };

  const handleMouseDown = (e) => {
    if (annotationMode === "positive") {
      setPositiveKeypoints((points) => [
        ...points,
        e.target.getStage().getPointerPosition(),
      ]);
      return;
    }

    if (annotationMode === "negative") {
      setNegativeKeypoints((points) => [
        ...points,
        e.target.getStage().getPointerPosition(),
      ]);
      return;
    }

    setDrawing(true);
    const pos = e.target.getStage().getPointerPosition();
    setRect({ x: pos.x, y: pos.y, width: 10, height: 10 });
  };

  const handleMouseMove = (e) => {
    if (!drawing) return;
    const pos = e.target.getStage().getPointerPosition();
    setRect((rect) => ({
      ...rect,
      width: pos.x - rect.x,
      height: pos.y - rect.y,
    }));
  };

  const handleMouseUp = () => {
    setDrawing(false);
  };

  const annotate = (event) => {
    setAnnotationIsLoading(true);
    const { x, y, width, height } = rect;
    axios
      .post("http://127.0.0.1:5000/api/annotate", {
        bbox: {
          x1: x / image.scale,
          y1: y / image.scale,
          x2: (x + width) / image.scale,
          y2: (y + height) / image.scale,
        },
        positiveKeypoints: positiveKeypoints.map((pos) => [
          pos.x / image.scale,
          pos.y / image.scale,
        ]),
        negativeKeypoints: negativeKeypoints.map((pos) => [
          pos.x / image.scale,
          pos.y / image.scale,
        ]),
        sceneName: currentScene?.sceneName,
      })
      .then((response) => {
        updateAvailableScenes();
      })
      .catch((error) => {
        console.error("Error annotating:", error);
        console.error("Error fetching coordinates:", error);
      })
      .finally(() => {
        setAnnotationIsLoading(false);
      });
  };

  const currentResult = currentScene?.result;

  const selectOptions = availableScenes
    .map((scene) => ({
      value: scene,
      label: scene.sceneName,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));

  const annotationModeSelectOptions = ["positive", "negative", "box"].map(
    (mode) => ({
      value: mode,
      label: mode.charAt(0).toUpperCase() + mode.slice(1),
    })
  );

  return (
    <>
      <Modal
        isOpen={isLoading}
        contentLabel="Loading Modal"
        ariaHideApp={false}
        style={{
          overlay: {
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
          },
          content: {
            position: "absolute",
            top: "50%",
            left: "50%",
            background: "none",
            border: "none",
            padding: "0",
          },
        }}
      >
        <ClipLoader color="#ffffff" loading={true} size={30} />
      </Modal>

      <div className="flex items-center justify-between p-4 bg-gray-800 text-white font-sans">
        <h1 className="text-4xl font-bold">SAM 2 Visualizer</h1>

        <div className="flex items-center justify-between space-x-4">
          <div className="flex items-center space-x-4">
            <button
              disabled={!rect}
              onClick={annotate}
              className={`font-bold py-2 px-4 rounded ${
                rect
                  ? "bg-blue-500 hover:bg-blue-700 text-white"
                  : "bg-gray-500 text-gray-200 cursor-not-allowed"
              }`}
              title={
                rect ? "" : "Draw a new rectangle to create a new segmentation"
              }
            >
              Segment
            </button>
            <div className="flex items-center space-x-4">
              <div>Show Mask</div>
              <Switch
                onChange={() => {
                  setShowMask(!showMask);
                }}
                checked={showMask}
              />
            </div>
            <div>Scene</div>
            <Select
              value={selectOptions.find(
                (option) => option.value.sceneName === currentScene?.sceneName
              )}
              onChange={handleChange}
              options={selectOptions}
              placeholder="Select scene"
              styles={customStyles}
            />
            <div>Manual label mode</div>
            <Select
              value={annotationModeSelectOptions.find(
                (option) => option.value === annotationMode
              )}
              onChange={(selectedOption) =>
                setAnnotationMode(selectedOption.value)
              }
              options={annotationModeSelectOptions}
              placeholder="Select annotation mode"
              styles={customStyles}
            />
            <button
              className="px-2 py-2 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded"
              onClick={handleClearManualAnnotations}
            >
              Clear manual labels
            </button>

            <button
              className="px-4 py-2 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded"
              onClick={handleRefresh}
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {!!image && (
        <div >
          <Stage
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            width={image ? image.img.width * image.scale : 800}
            height={image ? image.img.height * image.scale : 600}
          >
            <Layer>
              {image && (
                <Image
                  image={image.img}
                  scaleX={image.scale}
                  scaleY={image.scale}
                />
              )}
              {maskImage && showMask && (
                <Image
                  image={maskImage.img}
                  opacity={0.3}
                  scaleX={maskImage.scale}
                  scaleY={maskImage.scale}
                  filters={[colorReplaceFilter]}
                  ref={maskImageRef}
                />
              )}
              {!!rect && <Rect {...rect} stroke="#007BFF" strokeWidth={4} />}
              {positiveKeypoints.map((pos, i) => (
                <Circle key={i} x={pos.x} y={pos.y} radius={5} fill="green" />
              ))}
              {negativeKeypoints.map((pos, i) => (
                <Circle key={i} x={pos.x} y={pos.y} radius={5} fill="red" />
              ))}
            </Layer>
          </Stage>
        </div>
      )}
    </>
  );
};

export default App;
