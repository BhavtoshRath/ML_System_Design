1. How can altering loss function address class imbalance?

Modifying the loss function is a common technique to mitigate this issue by adjusting how errors are penalized for different classes. 
Two commonly used loss functions: class-imbalanced loss and focal loss.

---

Choosing between **training from scratch** and **fine-tuning** depends on factors like data availability, computational resources, and task similarity. This guide outlines when to use each approach.  

---

## Pre-training vs Fine-tuning

### **Comparison Table**  

| Feature            | Training from Scratch | Fine-Tuning (Transfer Learning) |
|--------------------|----------------------|--------------------------------|
| **Data Requirement** | Large dataset needed | Small dataset sufficient |
| **Training Time**  | Long | Short |
| **Computational Cost** | High | Low to Medium |
| **Performance**    | High (if enough data) | High (with limited data) |
| **Pretrained Models** | Not used | Used |
| **Best for**       | Unique/new tasks | Similar tasks with less data |

---

## Distributed training

### **1️⃣ Key Differences Between Model and Data Parallelism**

| Feature | **Model Parallelism** 🧩 | **Data Parallelism** 📊 |
|----------|----------------|----------------|
| **Definition** | Splits the **model** across multiple devices. | Splits the **dataset** across multiple devices. |
| **When to Use?** | When the model is **too large to fit into a single GPU’s memory**. | When the dataset is **too large** and training needs to be accelerated. |
| **How It Works?** | Different GPUs handle **different parts of the model** (e.g., layers, submodules). | Each GPU gets a **copy of the entire model** but trains on different **data batches**. |
| **Communication Overhead** | **High**: GPUs must communicate intermediate activations. | **Moderate**: Synchronization of gradients across GPUs after backpropagation. |
| **Scalability** | **Limited**: Effective only for very **large models**. | **Highly Scalable**: Works well with many GPUs. |
| **Example Models** | GPT-4, LLaMA-65B, ViT-Huge (Vision Transformers). | ResNet, BERT, EfficientNet. |
| **Common Frameworks** | **Megatron-LM, DeepSpeed, TensorFlow Pipeline Strategy**. | **PyTorch DDP, Horovod, TensorFlow MirroredStrategy**. |

---
## What loss function should we choose?

### Choosing the Right Loss Function for Machine Learning

| **Problem Type** | **Common Use Cases** | **Recommended Loss Function** | **Why?** |
|-----------------|----------------------|------------------------------|----------|
| **Regression**  | Predicting house prices, temperature forecasting, stock prices | **Mean Squared Error (MSE)** | Penalizes large errors more, useful when extreme deviations matter. |
| **Regression**  | Robust regression, reducing outlier impact | **Mean Absolute Error (MAE)** | Treats all errors equally, less sensitive to outliers than MSE. |
| **Regression**  | Predicting financial risk, optimizing human perception-based metrics | **Huber Loss** | Combines MSE and MAE, robust to outliers. |
| **Regression**  | Predicting time-to-failure, event durations | **Log-Cosh Loss** | Similar to Huber but differentiable everywhere, making optimization easier. |
| **Binary Classification** | Spam detection, sentiment analysis (Yes/No) | **Binary Cross-Entropy (Log Loss)** | Measures how well predictions match probabilities, widely used in logistic regression. |
| **Multi-Class Classification** | Image classification, NLP text categorization | **Categorical Cross-Entropy** | Works well with one-hot encoded target labels. |
| **Multi-Class Classification** | Image classification with soft labels | **Sparse Categorical Cross-Entropy** | Optimized version of categorical cross-entropy when labels are integers instead of one-hot vectors. |
| **Multi-Label Classification** | Tagging multiple objects in an image | **Binary Cross-Entropy** | Evaluates each class independently, suitable for multi-label tasks. |
| **Imbalanced Classification** | Fraud detection, medical diagnosis | **Focal Loss** | Gives more weight to hard-to-classify samples, improving model focus on minority classes. |
| **Ordinal Classification** | Movie ratings, survey responses (Ordered categories) | **Ordinal Cross-Entropy** | Accounts for the relative ordering between categories. |
| **Sequence-to-Sequence (Seq2Seq)** | Machine translation, speech-to-text | **Cross-Entropy + Teacher Forcing** | Helps in training autoregressive models with ground-truth alignment. |
| **Reinforcement Learning (Policy Gradient Methods)** | Robotics, game playing, autonomous driving | **Policy Gradient Loss (e.g., REINFORCE, PPO Loss)** | Maximizes the expected reward from an action sequence. |
| **Generative Models (GANs)** | Image synthesis, deepfake generation | **Adversarial Loss (Binary Cross-Entropy for GANs)** | Trains the generator to produce realistic samples against the discriminator. |
| **Contrastive Learning** | Face recognition, Siamese networks | **Contrastive Loss / Triplet Loss** | Maximizes similarity between similar samples while pushing apart dissimilar samples. |

---

## **💡 Tips for Choosing the Right Loss Function**
- **For regression:** MSE is standard, but use MAE or Huber Loss for robustness.  
- **For classification:** Use cross-entropy for most cases, but focal loss for imbalanced datasets.  
- **For structured prediction (e.g., ranking, NLP):** Consider task-specific losses like ordinal cross-entropy or seq2seq loss.  
- **For generative models & reinforcement learning:** Use adversarial loss or policy gradient-based losses.  
