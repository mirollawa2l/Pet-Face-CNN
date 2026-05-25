# Transfer Learning Details for Pretrained Models

## ResNet18

- **Weight Initialization:** Loads ImageNet weights.
- **Freezing Strategy:** Initially freezes all parameters in the network.
- **Fine-Tuning:** Unfreezes the last residual block (`layer4`), allowing adaptation of highest-level feature representations to pet facial expressions instead of relying strictly on ImageNet features.
- **Classifier Replacement:** Replaces the final fully connected layer (`fc`) with a new `nn.Sequential` block containing a `Dropout(0.3)` layer for regularization followed by the final `Linear` layer.

---

## DenseNet121

- **Weight Initialization:** Loads ImageNet weights.
- **Freezing Strategy:** Initially freezes all parameters in the network.
- **Fine-Tuning:** Unfreezes the final dense block (`features.denseblock4`) to fine-tune the model's most complex visual features.
- **Classifier Replacement:** Replaces the final classifier with a new `nn.Sequential` block containing a `Dropout(0.3)` layer and a new `Linear` layer.

---

## MobileNetV2

- **Weight Initialization:** Loads ImageNet weights.
- **Freezing Strategy:** Freezes the entire backbone; used as a static feature extractor (no convolutional blocks are unfrozen).
- **Classifier Replacement:** Targets the second element of the classifier block (`classifier[1]`) and replaces it with a new `Linear` layer configured for the target number of classes.

---

## VGG16

- **Weight Initialization:** Loads ImageNet weights.
- **Freezing Strategy:** Freezes the entire convolutional backbone (`features`).
- **Classifier Replacement:** Replaces only the final linear layer (`classifier[6]`) within the large sequential classifier block, leaving the pretrained intermediate dense layers and dropouts intact.

---

## InceptionV3

- **Data Preparation:** InceptionV3 requires larger input resolution. Use `INCEPTION_SIZE = 299` and specialized transforms/loaders for this model.
- **Weight Initialization:** Loads ImageNet weights (auxiliary logits enabled when constructing the model if needed).
- **Freezing Strategy:** Freezes the entire backbone.
- **Classifier Replacement:** Replace both the main final classifier (`fc`) and the auxiliary classifier (`AuxLogits.fc`) so both prediction heads output the correct number of classes.
- **Custom Training Loop:** Because of the auxiliary network, use a custom training function (e.g., `train_inception_transfer`) that handles two outputs. Compute the loss as a weighted sum: `loss = loss1 + 0.4 * loss2`, where `loss1` is the main output loss and `loss2` is the auxiliary output loss.

---

If you'd like, I can now insert these markdown sections directly into `cnn.ipynb` before each model's transfer-learning cell. Would you like me to do that (I can attempt precise cell insertion)?
