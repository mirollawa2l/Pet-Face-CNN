# CNN Models for Pet Facial Expression Classification

A PyTorch-based notebook project demonstrating convolutional neural network training, evaluation, and model comparison for pet facial expression recognition.

## What this project contains

- `cnn (1).ipynb` — a Jupyter notebook with a full training and evaluation workflow.
- Dataset download and organization using the Kaggle dataset `anshtanwar/pets-facial-expression-dataset`.
- Data augmentation and preprocessing for image classification.
- Multiple CNN model implementations:
  - VGG16
  - ResNet18
  - MobileNetV1
  - Inception-style architecture (Inception V3-inspired blocks)
- Training utilities with early stopping, learning rate scheduling, and metrics logging.
- Evaluation routines with accuracy, precision, recall, F1-score, classification report, and confusion matrix visualization.

## Key features

- Automated dataset download and local copy preparation.
- Image transforms for robust training, including random resized crop, horizontal flip, rotation, affine transforms, color jitter, and blur.
- Custom model definitions written from scratch.
- Clear visualizations for class distribution and sample augmented images.
- Model performance tracking and comparison.

## Usage

1. Open `cnn (1).ipynb` in Jupyter or VS Code.
2. Install required Python packages:
   - `torch`
   - `torchvision`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `kagglehub`
3. Run the notebook cells in order to download data, build models, train, and evaluate.

## Suggested improvements

- Add a dedicated `requirements.txt` or `environment.yml`.
- Refactor model classes and training functions into separate Python modules.
- Add command-line training scripts and saved model checkpoints.
- Experiment with pretrained backbones and transfer learning.

## Suggested project names

- `PetFaceCNN`
- `FelineFaceNet`
- `PawEmotionNet`
- `PetExpressionAI`
- `FurFaceClassifier`
- `PetMoodVision`
- `PawSenseCNN`
- `ExpressionPaws`
- `FacialFurNet`
- `PetEmotionVision`

## Notes

The repository currently contains a single notebook and can be expanded into a more reusable package with dedicated modules and scripts.
