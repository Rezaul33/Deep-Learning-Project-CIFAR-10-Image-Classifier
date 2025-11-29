# CIFAR-10 Image Classification Project

!CIFAR-10 Samples

## Overview
This project implements and compares different deep learning models for image classification on the CIFAR-10 dataset. The repository contains three main notebooks:

1. **DNN Implementation**: A baseline model using TensorFlow/Keras  
2. **Custom CNN**: A PyTorch implementation of a Convolutional Neural Network  
3. **ResNet-18**: A PyTorch implementation of the ResNet-18 architecture  

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

### Classes
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Models and Performance

### 1. Original Model (TensorFlow/Keras)
- **Architecture**: Custom CNN with multiple Conv2D and Dense layers  
- **Test Accuracy**: ~88.4%  
- **Training Time**: ~10 minutes on GPU  
- **Key Features**: Batch Normalization, Dropout, Data Augmentation  

### 2. Custom CNN (PyTorch)
- **Architecture**: Custom CNN with 3 convolutional blocks  
- **Test Accuracy**: ~93.3%  
- **Training Time**: ~15 minutes on GPU  
- **Key Features**: BatchNorm, MaxPooling, Dropout  

### 3. ResNet-18 (PyTorch)
- **Architecture**: Standard ResNet-18 with skip connections  
- **Test Accuracy**: ~93.3%  
- **Training Time**: ~30 minutes on GPU  
- **Key Features**: Residual blocks, BatchNorm, Learning Rate Scheduling  

## Model Comparison
| Model          | Test Accuracy | Training Time | Parameters | Best For                   |
|----------------|--------------|---------------|------------|----------------------------|
| Original (TF)  | 88.4%        | ~10 min       | ~1.2M      | Quick experiments          |
| Custom CNN     | 93.3%        | ~15 min       | ~1.8M      | Balance of speed/accuracy  |
| ResNet-18      | 93.3%        | ~30 min       | ~11.2M     | Best accuracy, transfer learning |

## Best Performing Model
The **ResNet-18** model achieved the best performance with **93.3% test accuracy**. It shows excellent generalization and handles class imbalance well.

### Key Strengths
- Handles vanishing gradient problem with skip connections  
- Generalizes well to unseen data  
- Can be fine-tuned for other image classification tasks  

## Project Structure
```
Deep-Learning-Project-CIFAR-10-Image-Classifier/
│
├── 1. EDA-CIFAR-10.ipynb # Data Exploration & Custom DNN implementation
├── 2. Model-Training-CNN.ipynb # Custom CNN implementation
├── 3. Model-Training-ResNet.ipynb # ResNet-18 implementation
├── Images/ # Sample images and visualizations
│ ├── cifar10_samples.png # Sample images from CIFAR-10
│ ├── cnn_architecture.png # CNN model architecture
│ ├── resnet_performance.png # ResNet training curves
│ ├── confusion_matrix.png # Model confusion matrix
│ └── training_curves.png # Training/validation metrics
├── License.txt
├── requirements.txt # Python dependencies
└── README.md # This file
```


## Installation

1. **Clone the repository**:
bash
```
git clone https://github.com/Rezaul33/Deep-Learning-Project-CIFAR-10-Image-Classifier
```
2. **Create and activate a virtual environment (recommended)**:
# Windows
```
python -m venv venv
.\venv\Scripts\activate
```
# Linux/Mac
```
python3 -m venv venv
source venv/bin/activate
```
3. **Install dependencies**:
```
pip install -r requirements.txt
```
4. **Launch Jupyter Notebook**:
```
jupyter notebook
```
## Run the Notebooks in Order

1. `EDA-CIFAR-10.ipynb`  
2. `Model-Training-CNN.ipynb`  
3. `Model-Training-ResNet.ipynb`  

## Usage

- Start with the EDA notebook to understand the dataset.  
- Run the CNN notebook to see a custom implementation.  
- Explore the ResNet notebook for state-of-the-art performance.  
- Modify hyperparameters or architectures as needed.  

## Results and Analysis

### Key Findings
- The ResNet-18 model achieved the highest accuracy of **93.3%** on the test set.  
- The custom CNN performed equally well but with fewer parameters.  
- Data augmentation significantly improved model generalization.  
- Learning rate scheduling helped in achieving better convergence.  

### Performance Metrics
- **Precision**: 93.3% (average across all classes)  
- **Recall**: 93.3% (average across all classes)  
- **F1-Score**: 93.3% (average across all classes)  

### Error Analysis
The model performs slightly worse on:  
- Cats (often confused with dogs)  
- Dogs (often confused with cats)  
- Birds (sometimes confused with airplanes)  

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.  
For major changes, please open an issue first to discuss what you would like to change.  

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.  

## Acknowledgments
- CIFAR-10 dataset  
- PyTorch and TensorFlow communities  
- Original ResNet paper: *"Deep Residual Learning for Image Recognition"* by Kaiming He et al.  

## Contact
- **Author**: Rezaul Islam  
- **GitHub**: [Rezaul33](https://github.com/Rezaul33)  
- **Email**: rezaul.islam.da@gmail.com  

## Additional Notes
- All models were trained on a single NVIDIA GPU.  
- Training times may vary based on hardware.  
- For best results, ensure you have a CUDA-compatible GPU for faster training.  

## Future Work
- Implement other architectures like EfficientNet or Vision Transformer.  
- Try advanced data augmentation techniques.  
- Implement model quantization for deployment.  
- Create a web demo for real-time classification.



