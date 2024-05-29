# Classification Image

---

## Drone Image Classification

### Project Overview

This project classifies images into two categories: drones and non-drones. It uses a Convolutional Neural Network (CNN) based on a pre-trained ResNet-18 model.

### Installation

Ensure you have Python installed along with the necessary libraries:

```bash
pip install torch torchvision numpy pillow matplotlib
```

### Usage

1. **Training the Model**:
   - Run `classify_image.py` to train the model.
   - The script performs data augmentation, loads a pre-trained ResNet-18, and trains it on your dataset.

2. **Classifying an Image**:
   - Use the trained model to classify new images.
   - The script loads the saved model, preprocesses the image, and predicts the class (drone or non-drone).

### Example

To classify an image:

```python
# After training the model, use the following script to classify an image:

# Load the saved model and classify a test image
image_path = '/path/to/your/test/image.png'
# The script loads and preprocesses the image, then prints the predicted class
```

### Model Accuracy

The model's accuracy is printed during training for both training and validation datasets.

---

### The Classify:


<img width="639" alt="Screenshot 2024-05-29 at 15 51 25" src="https://github.com/yoelchemla/classification_image/assets/74509202/ea03aeaf-da03-448d-9eec-0341b6bcaaac">
