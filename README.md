Image Segmentation using K-Means Clustering
Project Overview

This project demonstrates image segmentation using the K-Means clustering algorithm, an unsupervised machine learning technique.
The goal is to partition an image into different regions based on pixel color similarity, which is useful in computer vision tasks such as object detection, medical imaging, and image compression.

The implementation is done using Python, OpenCV, NumPy, Matplotlib, and scikit-learn.

 Objectives

1.Load and preprocess an image

2.Convert image pixels into a feature set

3.Apply K-Means clustering on pixel values

4.Segment the image into meaningful regions

5.Visualize the original and segmented images

 Technologies Used

-Python 3

-OpenCV (cv2)

-NumPy

-Matplotlib

-scikit-learn

-Jupyter Notebook

Image Segmentation/
│
├── image_segmentation.ipynb   # Main Jupyter Notebook
├── input_image.jpg            # Input image file
├── README.md                  # Project documentation

Dataset

-The project uses a single input image (input_image.jpg)
-Image is loaded from a local path using OpenCV


Methodology
1️ Image Loading

The image is read using OpenCV

Converted from BGR to RGB for correct visualization

2️ Data Preparation

Image pixels are reshaped into a 2D array

Each pixel is treated as a data point with 3 features (R, G, B)

3️ K-Means Clustering

The K-Means algorithm groups pixels into K clusters

Each cluster represents a dominant color region

4️ Image Segmentation

Each pixel is replaced with its cluster center color

Produces a segmented version of the original image

5️ Visualization

Original image and segmented image are displayed using Matplotlib



Machine Learning Technique Used
🔹 K-Means Clustering

Unsupervised learning algorithm

Groups similar data points based on distance

Uses Euclidean distance

Iteratively updates cluster centroids


Sample Output

-Original Image
-Segmented Image with reduced colors based on K clusters

Future Enhancements

-Automatic selection of optimal K
-Apply on grayscale images
-Use deep learning segmentation (CNN, U-Net)
-Build a GUI using Streamlit
-Apply to medical or satellite images


Author

Harini K
Department of Computer Science
