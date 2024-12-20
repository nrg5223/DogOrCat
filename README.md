# CNN Image Classifier: Dog or Cat
This repository contains a Jupyter Notebook implementing a Convolutional Neural Network (CNN) to classify images of cats and dogs. The project is built with Python and TensorFlow, demonstrating end-to-end steps from data preprocessing to model training, evaluation, and predictions.

## Project Overview
This project uses a CNN to classify 100x100 RGB images of cats and dogs. The notebook walks through:
1. **Data Preparation**: Loading and normalizing image data.
2. **Data Visualization**: Displaying samples from the training data.
3. **Model Architecture**: Building a CNN using TensorFlow's Keras API.
4. **Training the Model**: Training the model with binary cross-entropy loss function.
5. **Evaluation**: Testing the model's accuracy with the test set.
6. **Making Predictions**: Running the trained model on test samples and displaying the results.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- TensorFlow
Install the dependencies with:
```pip install numpy matplotlib tensorflow```

## Dataset
The dataset should include the following files:
- `data/input.csv`: Training image pixel data (flattened).
- `data/labels.csv`: Labels for the training data (0 for dog, 1 for cat).
- `data/input_test.csv`: Testing image pixel data (flattened).
- `data/labels_test.csv`: Labels for the test data.
Each Image in the dataset should be resized to 100x100 pixels and represented in RGB format.

## How to Run
1. Clone the repository:
```
git clone https://github.com/your-username/cnn-image-classifier.git
cd cnn-image-classifier
```
2. Place the dataset files in the `data` directory. They are too large to be put on GitHub, so find them here: https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD
3. Open the Jupyter Notebook:
```
jupyter notebook cnn_image_classifier.ipynb
```
4. Run the cells sequentially to train the model, evaluate it, and make predictions.

## Model Architecture
- **Input Layer**: 100x100 RGB images.
- **Convolutional Layers**: Two Conv2D layers with ReLU activation.
- **Pooling Layers**: Two MaxPooling2D layers for down-sampling.
- **Fully Connected Layers**:
    - 128 units with ReLU activation.
    - 1 unit with sigmoid activation (binary classification).
- **Optimizer**: Adam.
- **Loss Function**: Binary Cross-Entropy.

## Example Output
After running the notebook, the model predicts whether a given image is a cat or a dog with confidence scores. Example output:
```
Prediction: Cat
Confidence: 0.9876
```

## Further Improvement Options
- **Data Augmentation**: Improve generalization by augmenting training data.
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and layer configurations. With about an hour of playing around, I couldn't get the accuracy higher than ~72%. I think more data is most important for improving it further.
- **Transfer Learning**: Use pre-trained models for better performance on small datasets.
