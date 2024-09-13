# Face Recognition System using Transfer Learning

## Overview
This project implements a face recognition system using Transfer Learning with a pre-trained VGG16 model, along with OpenCV for face detection and real-time processing. It also includes data collection using a webcam for face sample generation and a custom classification system to identify individuals. The system leverages deep learning to classify faces from images or live video streams.

## Requirements
The following libraries are required to run the project:

- `tensorflow` for deep learning and model creation.
- `keras` for accessing pre-trained models and deep learning layers.
- `opencv-python` for real-time face detection and image processing.
- `matplotlib` for visualizing loss and accuracy plots.
- `numpy`, `pandas`, `glob`, `Pillow` for data handling and image processing.

Install the dependencies using:
```bash
pip install tensorflow keras opencv-python matplotlib numpy pandas glob pillow
```

## Project Structure
- **Face Recognition**: The model is trained to recognize specific faces using the VGG16 model, fine-tuned for classification with a softmax output layer. 
- **Data Collection**: The script collects face images using a webcam, crops them, and stores them in a specified directory for training.
- **Face Detection**: Haar Cascade classifiers are used for detecting faces and eyes from real-time video input.

## Data
The project requires two datasets:
1. **Training dataset**: Images for training the model, organized by folders representing classes (people).
2. **Test dataset**: Images for testing the model’s performance.

Both datasets are passed through data augmentation for better generalization.

## How to Run the Code
### 1. Train the Model
- Set the `train_path` and `test_path` to your datasets.
- Run the script to load the pre-trained VGG16 model, freeze the layers, and train on your dataset.
- The model will be saved as `my_own_model.h5` for later use.

### 2. Data Collection using Webcam
- Run the `cap = cv2.VideoCapture(0)` block to enable webcam and collect face samples. 
- Images will be stored in a folder for future training.

### 3. Face Recognition with Webcam
- The real-time face recognition system can be tested using live video. 
- The script will detect faces in real time and predict the identity based on the trained model.

### 4. Face and Eye Detection
- The system can detect both faces and eyes using pre-trained HaarCascade classifiers. 
- Real-time video detection will highlight faces and eyes using bounding boxes.

## Usage Instructions
1. **Training**: 
   - Customize the folder paths for the training and test datasets.
   - Train the model by running the script, and the results (accuracy, loss) will be visualized.

2. **Real-time Prediction**:
   - Load the saved model (`my_own_model.h5`), and the system will predict the face in real-time video using the webcam.

3. **Face Collection**:
   - Execute the script for webcam data collection and save images in the specified folder for further use.

4. **Face & Eye Detection**:
   - Run the detection code to highlight faces and eyes in real-time video.

## Model Architecture
- Pre-trained VGG16 used as the feature extractor.
- A dense layer with a softmax activation is used for multi-class classification.
  
## Example
For face prediction, once the model is trained:
```python
custom_image_path = r"your_image_path.jpg"
processed_image = preprocess_image(custom_image_path)
predictions = model.predict(processed_image)
predicted_class_label = class_labels[np.argmax(predictions)]
```

## Results
- The face recognition model achieved **96.88% accuracy** on the test set.
- Data augmentation improved generalization.
  
## Future Improvements
- Implement more advanced pre-processing techniques.
- Increase the dataset size for more accurate predictions.
  
## License
This project is licensed under the MIT License.

## Credits
- VGG16 pre-trained model provided by `Keras`.
- Face detection using HaarCascade classifiers from `OpenCV`.
