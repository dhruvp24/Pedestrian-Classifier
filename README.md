# Note
https://purdue0-my.sharepoint.com/:f:/g/personal/pate2372_purdue_edu/EtOebKVmOI9LtpfPtWFRvb8B0eD4NcQ4ZlEg0KILXFLPgw?e=t7mJd4
Link to onedrive for additional bigger size file.

# Pedestrian Classification Using Deep Learning

## Description

### Inspiration

Last summer, I worked on a project where we manually annotated video datasets that featured either e-scooter riders or bicycle riders. While the task itself wasn’t particularly challenging, it quickly became overwhelming due to the sheer size of the datasets. Each video required a human reviewer to watch and record timestamps, a process that was both time-consuming and error-prone. This experience made me realize how valuable an AI-powered classifier could be, not only to speed up the annotation process but also to reduce human error and free up valuable time for other tasks.

### What It Does

The goal of this project is straightforward yet impactful. The trained deep learning model takes an image as input and classifies it into one of three categories: pedestrian, e-scooter rider, or bicycle rider. This automation can significantly streamline processes that traditionally rely on human intervention, such as annotating video datasets for autonomous driving or transportation safety projects.

### How I Built It

To build this classifier, I used PyTorch and employed transfer learning to fine-tune a pre-trained ResNet-18 model for image classification tasks. Transfer learning helped leverage the power of a pre-trained network, saving time and computational resources. I retrained the final layers of the ResNet-18 model on a dataset of labeled images, ensuring the model could effectively distinguish between pedestrians, e-scooter riders, and bicycle riders. Once the classifier is ready, we can use YOLO to detect the person class in any image and then use our classifier on the detected bounding box to determine if the person is a pedestrian, e-scooter rider or bicycle rider.

### Challenges We Ran Into

Training the model posed a few challenges, primarily related to processing power and time. Image classification models, particularly deep learning models like ResNet-18, require significant computational resources. Even with access to Kaggle’s GPU resources, fine-tuning the model took approximately two hours to complete. Balancing the need for high-quality data and training efficiency was a learning experience that involved fine-tuning hyperparameters and ensuring that the dataset was well-representative of real-world scenarios.

### What's Next for Pedestrian Classification Using Deep Learning

While the current model performs well for single-image classification, there's potential to expand its capabilities. One avenue for growth would be to develop additional classifiers for other important road objects, such as traffic lights and traffic signs. Integrating these specialized classifiers with an object detection system like YOLO could enable comprehensive detection and classification of various road objects in real-time video feeds. This would be a critical step toward automating the annotation process for datasets used in autonomous driving and pedestrian path prediction, providing a much more efficient and scalable solution for researchers and engineers.

## Files and folders

1. Pedestrian_classifier.ipynb - This was used to train the classifier on kaggle
2. Inference.ipynb - User can give their own input to check how the classifier works
3. Sample Inputs - This folder contains sample images which can be used as input to test the model
4. Results - This folder contains the output the inferenence.ipynb genrates when the sample inputs are used.
5. Models (present in One Drive) - This folder contains the trained models which can be loaded in the inferenece.ipynb
6. Demo - This video file shows how to use inference.ipynb on the kaggle to check the model

## Datasets

1. E-scooter rider and pedestrain dataset: http://escooterdataset.situated-intent.net/
2. Cyclist Dataset for Object Detection: https://www.kaggle.com/datasets/semiemptyglass/cyclist-dataset

## Data preprocessing
The E-scooter rider dataset had around 10k images of e-scooter rider and 10k images for pedestrian.

The cyclist dataset came as seperate full images and it's bounding box labels. Since, I only needed the cyclists from the full images, I cropped them out using the lable information. This generated around 22k images. I randomly chose 10k images since I wanted the dataset to be balanced.


