# MIBenchmarking: BCI Performance Evaluation Framework

A lightweight, modular framework developed for benchmarking Motor Imagery (MI) signals. This repository is optimized for high-performance execution on lab servers and local workstations, focusing on efficient data processing and model training for Brain-Computer Interface applications.

## Overview

This project serves as the core computational framework for my thesis research. It provides a standardized pipeline for:

* **System Diagnostics:** Hardware and environment verification via `spec_checker.py`.
* **Signal Processing:** Custom EEG filtering and artifact removal in `filters.py`.
* **Model Training:** End-to-end BCI training routines implemented in `train_bci.py`.

## Lab Server Deployment

This repository is designed for "headless" operation. Follow these steps to set up the environment on a remote server:

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Elric-dev/MIBenchmarking.git
cd MIBenchmarking

# Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
# Ensure you have a requirements.txt file (see 'Next Steps' below)
pip install -r requirements.txt

```

### 2. Verify Server Specifications

Before running heavy training jobs, verify that the server has the necessary resources (CPU/GPU/RAM):

```bash
python spec_checker.py

```

### 3. Execution

To run the main benchmarking or training pipeline:

```bash
python train_bci.py

```

## Project Structure

* `main.py`: Entry point for the benchmarking suite.
* `train_bci.py`: Contains the logic for training and evaluating BCI classifiers.
* `filters.py`: Signal processing modules (Bandpass, Notch, etc.) for EEG data.
* `spec_checker.py`: Utility script to log hardware specs and library versions for reproducibility.

## Key Features

* **Memory Efficient:** Optimized data loading to handle high-density EEG datasets without exceeding server RAM.
* **Modular Design:** Easily swap out filtering methods or model architectures for different benchmarking scenarios.
* **Reproducibility:** Integrated logging to ensure all experiment parameters are recorded.

## License
MIT License

```
