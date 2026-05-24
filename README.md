# CNN Models for Pet Facial Expression Classification

A PyTorch-based notebook project that trains and compares convolutional neural networks on pet facial expression data.

## Project structure

- `cnn.ipynb` — the main notebook containing the full workflow.
- `README.md` — explains project goals, code structure, and how to use the notebook.

## What this project does

The notebook performs a complete image classification pipeline for a pet facial expression dataset:

1. Downloads the dataset automatically from Kaggle using `kagglehub`.
2. Applies reproducible setup and seeding for deterministic training behavior.
3. Defines robust augmentation and normalization pipelines for training and evaluation.
4. Loads train/validation/test datasets using `torchvision.datasets.ImageFolder`.
5. Builds multiple CNN architectures from scratch.
6. Trains models with early stopping and learning rate scheduling.
7. Evaluates models using accuracy, precision, recall, F1 score, and confusion matrices.
8. Compares models trained from scratch and optionally transfer learning models.

## Code explanation

### Notebook sections

- **Dataset download**: Uses `kagglehub.dataset_download` to fetch the `anshtanwar/pets-facial-expression-dataset`, then copies it into a local `dataset` folder.
- **Environment setup**: Imports required libraries and prints the available device (`cuda` or `cpu`).
- **Reproducibility**: Sets seeds for Python, NumPy, and PyTorch and configures deterministic CuDNN behavior if a GPU is available.
- **Data transforms**:
  - `train_transform` includes random resized crop, horizontal flip, rotation, affine perturbations, color jitter, and random blur.
  - `eval_transform` resizes images and normalizes them with ImageNet-style mean and standard deviation.
- **Data loading**:
  - `ImageFolder` is used for the train, validation, and test splits.
  - `DataLoader` creates batched iterators with shuffling for training and deterministic ordering for validation/test.
- **Dataset analysis**:
  - Plots class distribution for the training split.
  - Visualizes augmented image samples for each class.
  - Prints per-split class counts.
- **Training utilities**:
  - `train_model()` runs each epoch, computes loss, evaluates on validation data, and applies early stopping.
  - `evaluate_model()` computes final metrics on the test set and plots a confusion matrix.
  - `plot_history()` visualizes training/validation loss and validation accuracy over epochs.

### Models implemented

- **VGG16**: A deep, sequential CNN using stacked 3×3 convolutions and max pooling. It is a classic baseline that emphasizes simplicity and depth.
- **ResNet18**: Adds residual skip connections to improve gradient flow and training stability. ResNets are effective for deeper networks and help prevent vanishing gradients.
- **MobileNetV1**: Uses depthwise separable convolutions to greatly reduce parameter count and compute cost. This model is designed for efficiency and is useful for mobile-style deployments.
- **Inception V3-inspired architecture**: Implements Inception-style modules with parallel branches and factorized convolutions to capture multi-scale features.
- **DenseNet121-inspired architecture**: Uses dense connectivity where each layer receives inputs from all previous layers, promoting feature reuse and efficient gradient flow.

### Training and evaluation flow

- The notebook configures hyperparameters such as epochs, learning rate, batch size, and early stopping patience.
- Each model is trained using Adam optimizer, cross-entropy loss, and a learning rate scheduler that reduces the learning rate on plateau.
- The best weights are restored after early stopping based on validation loss.
- Final test metrics are printed and stored for comparison.

### Transfer learning

The notebook also includes an optional transfer learning experiment using pretrained models from `torchvision.models`:

- ResNet18
- DenseNet121
- MobileNetV2
- VGG16
- InceptionV3

These models are adapted by freezing backbone parameters and replacing the classifier layers with new output heads for the dataset classes.

## Dependencies

Install the following packages in your Python environment:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kagglehub`

## How to run

1. Open `cnn.ipynb` in Jupyter or VS Code.
2. Ensure dependencies are installed.
3. Execute cells from top to bottom.

## Suggested improvements

- Add `requirements.txt` or `environment.yml`.
- Break the notebook into reusable Python modules, such as `data.py`, `models.py`, and `train.py`.
- Save model checkpoints and log experiments.
- Add a command-line interface for training and evaluation.
- Add a README section summarizing results once models are run.
