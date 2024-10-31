![Screenshot from 2024-10-31 06-25-44](https://github.com/user-attachments/assets/09432f40-6e00-4fff-bb94-76b7001faf9c)

# SAM 2 Visualizer 🎯
An interactive visualization tool for Facebook's [Segment-Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) model. This tool allows you to easily visualize and experiment with SAM2's segmentation capabilities through an intuitive user interface.

The service is divided into two parts:
- **Backend:** a Flask server that runs the SAM2 model
- **Frontend:** a React app that provides an interactive interface for image segmentation

## Features 🌟
- Interactive image selection and visualization
- Multiple annotation modes (box, positive points, negative points)
- Real-time segmentation preview
- Adjustable mask overlay
- Support for multiple images

## Usage 📖

Complete the [Installation 🔧](#installation-🔧) first.

### Starting the backend
To start the backend, run the following command in the `backend` directory:

`python server.py ../dataset/ ../sam2/checkpoints/sam2.1_hiera_large.pt`

where `../dataset` is the path to your images directory.

This will start the server at [http://localhost:5000](http://localhost:5000)

### Starting the frontend
To start the frontend, run the following command in the `frontend` directory:

`yarn start`

This will start the UI at [http://localhost:3000/](http://localhost:3000/)

In the interface, you can:
1. Select an image from the dropdown
2. Choose an annotation mode (box, positive points, negative points)
3. Click and drag to create a bounding box or add points
4. Click "Segment" to generate the mask
5. Toggle mask visibility with the "Show Mask" switch

## Installation 🔧

### Backend Installation
First, create a Python environment (conda recommended).
Then follow these steps:

1. Install the [Segment-Anything](https://github.com/facebookresearch/segment-anything) repository as a submodule: `git submodule update --init --recursive`

2. Download the SAM2 model weights: `./download_sam_weights.sh`


3. Install the requirements: `pip install -r backend/requirements.txt`


### Frontend Installation
Follow these steps in the `frontend` directory:
1. Install `node` (for example with [nvm](https://github.com/nvm-sh/nvm))
2. Install `yarn` (for example with [npm](https://classic.yarnpkg.com/lang/en/docs/install/#debian-stable))
3. Run `yarn` to install the dependencies