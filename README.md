# Indian-Traffic-sign-Classification
A model capable of accurately identifying and classifying Indian traffic signs from real-world images.


Indian Traffic Sign Classification
This project develops a deep learning model to accurately identify and classify Indian traffic signs from real-world images. The model is integrated into a live Gradio web interface for real-time predictions.

🎯 Objective
The objective of this project is to develop a model capable of accurately identifying and classifying Indian traffic signs from real-world images. The system will assist in enhancing driver assistance systems (ADAS), autonomous vehicle navigation, and traffic management by recognizing regulatory, warning, and informational signs in Indian road conditions.

Problem Statement
India has a diverse and complex road network with varying environmental conditions and signboard quality. Traffic signs often differ in size, color contrast, illumination, and visibility due to dust, damage, or poor maintenance. Existing models trained on foreign datasets (like GTSRB) fail to generalize well to Indian signs. Therefore, a specialized deep learning model trained on Indian traffic sign datasets is required to achieve reliable and real-time classification performance without relying on OpenCV-based preprocessing.

Dataset
The model is trained on the Indian-Traffic-Sign-Classification dataset from Hugging Face.

Classes: 85

Splits: The dataset contains a single train split, which I manually divided into training and validation sets.

🛠 Tech Stack
Python 3.10+

TensorFlow (Keras) for model building and training.

Hugging Face datasets for loading and preprocessing data.

Gradio for building the interactive web interface.

PIL (Pillow) for image manipulation.

Journey & Methodology
The path to a final model involved several key iterations and failures. A significant personal challenge in this project was the initial hurdle of understanding and implementing CNNs. While the theory of convolutional layers, filters, and max pooling was clear on paper, translating it into a working, high-performance Keras model was a major difficulty.


1. Initial Attempt: Custom CNN (from Scratch)
Approach: I first built a simple 3-layer Convolutional Neural Network (CNN).

Result: Failure. The model failed to learn, with accuracy stuck at ~5%. The loss and accuracy graphs were flat, indicating the model was underfitting and was not complex enough to capture the features of 85 different classes.

2. Second Attempt: Deeper Custom CNN
Approach: Built a deeper, VGG-style model with more convolutional blocks.

Result: Failure. This model also failed, producing the same ~5% accuracy. This confirmed that training a model "from scratch" on this dataset is extremely difficult and requires complex architectures and tuning.

3. Final Solution: Transfer Learning (ResNet50)
This was the breakthrough. Instead of teaching a model to see edges and shapes from scratch, I used a model that already knew how.

Approach: Implemented Transfer Learning using the ResNet50 model, which was pre-trained on the massive ImageNet dataset.

Phase 1 (Head Training): "froze" the pre-trained ResNet layers and added our own "classifier head" (a GlobalAveragePooling2D layer and a Dense layer for 85 classes). We trained only this new head.

Phase 2 (Fine-Tuning): To improve performance, "unfroze" the top 20 layers of the ResNet model and continued training with a very small learning rate. This allowed the model to fine-tune its pre-trained knowledge specifically for Indian traffic signs.

Result: Success. This two-phase approach dramatically improved performance, achieving a final validation accuracy of 75.57%.

🚧 Major Difficulties & Solutions
This project was a case study in debugging deep learning models.

Difficulty: Extreme Underfitting

Problem: custom CNNs couldn't get above 5% accuracy.

Solution: pivoted to Transfer Learning. This is the key takeaway: for most image tasks, a pre-trained model like ResNet, VGG, or EfficientNet is the correct starting point.

Difficulty: Overfitting with ResNet

Problem: first ResNet model (before fine-tuning) hit a plateau around 72-74%. The training graphs showed training accuracy rising while validation accuracy went flat and validation loss began to increase—a classic sign of overfitting.

Solution: used two Keras Callbacks:

EarlyStopping(restore_best_weights=True): This automatically monitored the val_loss and stopped training when the model started to overfit, restoring the weights from the single best-performing epoch.

ReduceLROnPlateau: This automatically lowered the learning rate when the model hit a plateau, helping it find a more precise solution.

Difficulty: Gradio Interface "Error"

Problem: The deployed Gradio app showed a generic "Error" box when an image was uploaded.

Solution: This was a silent bug. The hard-coded list of CLASS_NAMES in the app was in a different order than the one the model was trained on. The fix was to programmatically load the class names directly from the Hugging Face datasets object, ensuring the app's list and the model's training list were identical.

📈 Final Results
Model: A ResNet50 model, fine-tuned for the Indian Traffic Sign dataset.

Final Accuracy: 75.57%

Interface: A working Gradio web app for live classification.

