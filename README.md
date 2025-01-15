# IAM Letter Recognition Project

This project aims to recognize letters from the IAM dataset using a neural network model. The IAM dataset contains handwritten text samples, which are used to train and evaluate the model's performance in recognizing individual letters.

## Project Structure

```
iam-letter-recognition
├── data
│   ├── raw                # Contains the raw IAM dataset files
│   └── processed          # Stores the processed data after preprocessing steps
├── models
│   └── model.py          # Defines the neural network model architecture
├── notebooks
│   └── exploration.ipynb  # Jupyter notebook for exploratory data analysis
├── src
│   ├── data_preprocessing.py  # Functions for loading and preprocessing the dataset
│   ├── train.py               # Responsible for training the neural network model
│   └── evaluate.py            # Evaluates the performance of the trained model
├── requirements.txt          # Lists the dependencies required for the project
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd iam-letter-recognition
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the IAM dataset and place the raw files in the `data/raw` directory.

## Usage

- To preprocess the data, run:
  ```
  python src/data_preprocessing.py
  ```

- To train the model, execute:
  ```
  python src/train.py
  ```

- To evaluate the model's performance, use:
  ```
  python src/evaluate.py
  ```

- For exploratory data analysis, open the Jupyter notebook:
  ```
  jupyter notebook notebooks/exploration.ipynb
  ```

## Overview of the IAM Dataset

The IAM dataset is a collection of handwritten text samples, which includes various styles and forms of handwriting. It is widely used for training and evaluating handwriting recognition systems. This project focuses on recognizing individual letters from the dataset, contributing to advancements in optical character recognition (OCR) technologies.