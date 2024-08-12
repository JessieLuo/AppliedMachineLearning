# Detecting Fake News with a BERT Model

## Implementation Idea

### Processing steps

1. load the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Then, obtain the [pre-trained BERT](https://huggingface.co/google-bert/bert-base-uncased) model to use as the base of the Fake News Detection Model using HuggingFace’s Transformers library.
3. Define the Base Model and the overall architecture. Apply PyTorch for defining, training, and evaluating the neural network models
4. Use freezing all the weights idea on the starting layers from BERT. If don’t do this, we lose all of the previous learning.
5. Then create new Trainable Layers. Generally, feature extraction layers are the only knowledge could be reused from the base model. To predict the model’s specialized tasks, it must add additional layers on top of them.
6. Additionally, define a new output layer, as the final output of the pre-trained model will almost certainly differ from the output we want for our model, which is binary 0 and 1
7. As the last step, fine tune the model.
8. And once all done, we move on to make predictions using our Fake News Detection Model on unseen data

### Model Architecture

A custom model architecture is defined by extending the ```torch.nn.Module``` class. The architecture includes:

- A pre-trained BERT model.
- Two fully connected layers (fc1 and fc2) with a ReLU activation function and dropout for regularization.
- A LogSoftmax function for the final output layer, which is suitable for classification tasks.

### Training Process

- Forward pass: Input sequences and attention masks are fed into the model.
- Loss Calculation: The Cross-Entropy Loss is used, which measures the discrepancy between predicted class probabilities and the actual labels.
- Backward pass: Gradients are calculated, and the model parameters are updated using an optimizer like Adam.

### Evaluation

The model's performance is evaluated on the validation and test sets using metrics, including accuracy, precision, recall, and F1 score. The Confusion Matrix is also used to visualize the classification results.
