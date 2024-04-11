![Project Header](/figures/project_header.png)

# Fire Detection

## Abstract

Forest fires represent an important threat to natural ecosystems. Early
detection is essential to prevent extensive damage and reduce risks associated with them. Conventional
fire detection systems primarily rely on smoke or temperature-based sensors. These approaches have inherent limitations that restrict their effectiveness,
particularly in outdoor environments and forested areas.

Our approach attempts to combine the feed obtained from traditional video or surveillance cameras with
a motion detection algorithm and a model to predict forest fires in real-time. This approach is
more cost-effective and requires less human intervention than other methods.

## Implementation

We use a Motion Detection Algorithm that uses a **background subtraction** method with **recursive updates** of thresholds and estimated backgrounds.

![Motion Detection Algorithm](/figures/motion_detection_algorithm.png)

The used model is an **XGBoost model** pre-trained on labeled image datasets containing fire and non-fire images. A histogram was computed for the color channels to differentiate without using the entire images.

![Model Training](/figures/model_training.png)

Additional details can be found in the [Technical Report](/reports/Technical_Report.pdf) and in the [Project Slides](/reports/Project_Slides.pdf).

## Installation

First, create a new `conda` environment using:

```sh
conda env create -f environment.yml
```

Activate the new environment using:

```sh
conda activate FireDetection
```

## Usage

The file `main.py` in the `src` directory contains a simple interface for using the program.

To execute, use:

```sh
python ./src/main.py
```

To change the video that is being used for detection, replace the `video_path` value in line 18 of `main.py` with the path to your target video.

Additionally, the algorithm can be tested with a live feed from http://66.119.104.155/mjpg/video.mjpg.

The example video was sourced from: [Fire: Fountaingrove in Santa Rosa (Monday, Oct. 9)](https://www.youtube.com/watch?v=TR-9IdfqaKY)

## Authors

:link: [Alessio Olivieri](https://github.com/Alessio-Olivieri)

:link: [Robert Li](https://github.com/mediolanum1)

:link: [Emilio Soriano Chávez](https://github.com/ami-sc)

This project was made as part of the 2023 AI Lab: Computer Vision and NLP course at Sapienza Università di Roma.
