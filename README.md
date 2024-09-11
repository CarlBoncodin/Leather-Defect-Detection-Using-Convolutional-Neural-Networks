# Leather Defect Detection Using Convolutional Neural Networks

Shoes have been a vital component of human protection and comfort throughout history. The footwear industry is growing steadily, with an annual growth rate of 3.43% and a projected revenue of USD 145 billion (Statista, 2023). Quality remains a critical factor influencing consumer purchases, but current quality assessment processes, which are largely manual, struggle with efficiency and accuracy, especially when handling large quantities (V-Trust, 2023). This project addresses these challenges by proposing the use of a generated dataset and training a RetinaNet model with the YOLO v8 Backbone architecture to detect leather defects. The goal is to enhance the efficiency and accuracy of quality assessment in the footwear industry, with a focus on applications in the Philippines.

Objectives
-
- Develop a convolutional neural network for detecting surface defects in leather.
- Utilize RetinaNet with YOLO v8 Backbone for model training.
- Employ a dataset annotated with leather defects for training and evaluation.
- Implement and optimize the model's architecture through hyperparameter tuning to enhance performance.

Methodology
-
Dataset Preparation
- The project uses a leather defect detection dataset from Roboflow by Renz (2022), consisting of 1259 images with 2436 defect annotations across three classes: stain, cut, and fold. The images are 416x416 pixels in size. The dataset includes 51 test images with 108 object instances, and additional test images from a shoe manufacturing company. Metadata is provided in a .csv format for parsing into a TensorFlow Dataset.

Model Architecture

- YOLOv8: Features a backbone network with convolutional layers and down-sampling operations to extract features, a feature pyramid network for multi-scale feature maps, and a detection head for bounding box coordinates and class predictions.
- RetinaNet with YOLOv8 Backbone: Utilizes the YOLOv8 backbone for feature extraction, followed by a feature pyramid network and detection head to predict bounding boxes, objectness scores, and class probabilities.

Hyperparameters
- Key hyperparameters include backbone architecture, learning rate, batch size, optimizer, loss function, and number of epochs. Early stopping is used to prevent overfitting.

Data Augmentation
- Techniques such as random flipping, jittered size scaling, shearing, and adjusting brightness and contrast are applied to increase dataset variability and robustness.

Different Model Backbone Testing
- Models are tested with various backbones, including YOLOv8 Large and RetinaNet with ResNet and YOLOv8 backbones. The performance of each model is evaluated to determine the best architecture and backbone for defect detection.

Learning Rate and Batch Size Modification
- The learning rate and batch size are adjusted to explore their effects on model performance. Batch sizes of 10 and 100, and learning rates of 0.001 and 0.0005, are tested to optimize convergence and generalization.

Results
- 

Model Performance

- YOLOv8 Detector with YOLOv8 Large Backbone: Achieved 76.23% accuracy on the test dataset. Accuracy decreased to 60% on the augmented dataset, indicating that data augmentation may not have been effective.
- RetinaNet with ResNet Backbone: Failed to produce meaningful predictions, indicating that this backbone was not suitable for the task.
- RetinaNet with YOLOv8 Large and XLarge Backbones: Achieved accuracies of 76.61% and 76.61% respectively. The large backbone performed better on the test dataset compared to the augmented dataset, with an accuracy of 75%. Data augmentation again did not improve performance.
- Learning Rate and Batch Size Modification: The best performance was obtained with RetinaNet using the Large YOLOv8 Backbone, with an accuracy of 80% after adjusting the batch size to 100 and the learning rate to 0.0005.

Sample Predictions
- Model predictions varied across different configurations. The RetinaNet with Large YOLOv8 Backbone showed the most promising results, although not all defects were accurately classified.

The project successfully developed a RetinaNet model with a YOLO v8 backbone for detecting leather defects. Using TensorFlow and Keras, the model was trained with various hyperparameter settings and backbone architectures. The highest accuracy was achieved with the RetinaNet model using the Large YOLOv8 Backbone, reaching 81%. Data augmentation was found to be detrimental to model performance, suggesting that it was not suitable for this specific application. Hyperparameter tuning, especially adjustments to learning rate and batch size, played a significant role in improving accuracy. Transfer learning was crucial for efficient model training and achieving favorable results.


