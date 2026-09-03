# **FlexBioNeuro Project**

Welcome to the FlexBioNeuro project repository! This funded project aim at optimizing the analysis of state parameters within biogas plants, specifically focusing on the anaerobic digestion process in fermenters.

## Background

Inside the fermenter of a biogas plant, numerous chemical reactions occur during the anaerobic digestion process. Traditional methods involve manual sampling and subsequent laboratory analysis to measure parameters such as acetic acid, VFA/TA ratio, and dry matter content. Unfortunately, this process is time-consuming and resource-intensive, primarily due to the absence of real-time measurement devices for specific parameters.

## Objective

The goal of the FlexBioNeuro project is to leverage the insights from literature suggesting that Near-Infrared (NIR) sensors can predict state parameters of biogas plants. This repository contains data collected from the biogas plant in Grub, Germany, over approximately one year. The collected samples underwent laboratory preparation, NIR sensor measurement, and analysis with GC, titrator, and moisture analyzer to determine acetic acid concentration, VFA/TA ratio, and dry matter content.

## Repository Structure

- **Data:** Contains the measured data, split into training, validation, and test sets. This is the single copy for the whole repository; scripts must not keep their own.

- **paths.py:** Resolves every dataset file through `data_path("<filename>.csv")` / `calibration_path()`, anchored to the repository root rather than the current working directory. This means a script produces the same result whether it is launched from the repository root, from its own directory, or by Slurm. Set the `NIR_DATA_DIR` environment variable to read the dataset from elsewhere (for example a cluster scratch filesystem) without editing any script.
  
- **Classification and Regression:** Focuses on the estimation of acetic acid concentration and VFA/TA ratio as a classification problem, and dry matter content as a regression problem.

  - **choose hyperparameter:** Subdirectory for finding the optimal hyperparameters using Group Stratified 10-Fold cross-validation. Utilizes either grid search or metaheuristic algorithms (Genetic Algorithm and Particle Swarm Optimization).

  - **evaluate model with new data set:** Subdirectory where the optimal hyperparameters are applied to train the entire dataset. The model is then evaluated against a separate test set that was not involved in the hyperparameter selection.

**Note:** The training process may be time-intensive, as it was conducted on a Linux cluster over a duration of 72-96 hours.

## Hardware Components

Hardware.txt: This file lists all the NIR-sensors required for the project.

## 3D Schematics

3D Schematics Directory: This directory contains all the developed and 3D printed components used in this project. The 3D designs are provided in both STEP and STL formats, allowing for easy constumization and direct 3D printing of the constructions developed.

## Contribution

We hope this repository contributes to advancements in real-time monitoring and optimization of biogas plant processes. Feel free to explore and contribute!
