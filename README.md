# Place Recognition Using Convolutional Neural Networks (VGG16)

## Overview
This project explores **place recognition** using **Convolutional Neural Networks (CNNs)**, particularly the **VGG16 architecture**. The model is trained and evaluated on a dataset of 200 garden images to match query images with reference images by utilizing deep learning techniques. The project focuses on image classification, distance metrics, and precision-recall analysis, providing insights into the effectiveness of CNNs for place recognition in garden environments.

## Features
- **VGG16-based CNN**: Utilizes a pre-trained VGG16 model for extracting deep features from images.
- **Precision-Recall Curve Analysis**: Visualizes the trade-off between precision and recall at different thresholds.
- **Cosine Similarity & Euclidean Distance**: Employed to measure image similarity based on feature vectors.
- **Keras Implementation**: The entire workflow is implemented in Keras with TensorFlow backend.

## Requirements
To run the project, ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

The main packages required are:
- `TensorFlow` / `Keras` (Deep Learning Framework)
- `Scikit-learn` (Evaluation Metrics)
- `Matplotlib`, `Seaborn` (Visualization)
- `NumPy`, `Pandas` (Data Handling)

## Dataset
The dataset consists of **200 images** captured from diverse garden environments, split into:
1. **Training Dataset**: 400 images (200 reference + 200 feature extraction images).
2. **Testing Dataset**: 200 images for comparison and matching.

## Model Architecture
The VGG16 architecture, pre-trained on ImageNet, serves as the backbone of the feature extraction process. Key components:
1. **Feature Extraction**: VGG16 layers are used to extract meaningful image representations.
2. **Distance Calculation**: Cosine similarity and Euclidean distance metrics are used to compare feature vectors and find the closest match.
3. **Classification**: The model classifies images based on their similarity to reference images.

## Usage

1. **Preprocessing**: Start by preprocessing the dataset:
   ```bash
   python preprocess.py --dataset your_dataset_directory
   ```

2. **Training the Model**: Train the VGG16 model for feature extraction:
   ```bash
   python train.py --dataset preprocessed_data_directory
   ```

3. **Image Matching**: After training, test the model with query images to match with reference images:
   ```bash
   python test.py --query_image query_image_path --reference_images reference_images_directory
   ```

4. **Precision-Recall Analysis**: Visualize the performance using precision-recall curves:
   ```bash
   python analyze.py --results results_directory
   ```

## Results
- **Validation Accuracy**: Achieved 86.25% validation accuracy by the 10th epoch.
- **Precision-Recall Curve**: Demonstrates the model’s ability to balance precision and recall across various thresholds.
- **Similarity Visualization**: Displays the top 5 most similar images from the reference dataset for a given query image.

## Visualization

- **Accuracy Curve**: Plots training and validation accuracy over epochs.
- **Precision-Recall Curve**: Visualizes the precision-recall trade-off for the classification model.
- **Similarity Score**: Displays cosine similarity and ranking of query images against reference images.

## Future Work
1. **Dataset Expansion**: Incorporating larger datasets from diverse environments to improve generalization.
2. **Model Optimization**: Exploring different CNN architectures (e.g., ResNet, Inception) to enhance performance.
3. **Real-World Deployment**: Testing the model’s utility in applications like autonomous navigation, augmented reality, and environmental monitoring.

## Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request with your enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README outlines the key aspects of your project based on the report and provides essential details for running and understanding the project. Let me know if you need any modifications!
