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
## What loss function should we choose?

### Choosing the Right Loss Function for Machine Learning

| **Problem Type** | **Common Use Cases** | **Recommended Loss Function** | **Why?** |
|-----------------|----------------------|------------------------------|----------|
| **Regression**  | Predicting house prices, temperature forecasting, stock prices | **Mean Squared Error (MSE)** | Penalizes large errors more, useful when extreme deviations matter. |
| **Regression**  | Robust regression, `reducing outlier impact` | **Mean Absolute Error (MAE)** | Treats all errors equally, less sensitive to outliers than MSE. |
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


### **💡 Tips for Choosing the Right Loss Function**
- **For regression:** MSE is standard, but use MAE or Huber Loss for robustness.  
- **For classification:** Use cross-entropy for most cases, but focal loss for imbalanced datasets.  
- **For structured prediction (e.g., ranking, NLP):** Consider task-specific losses like ordinal cross-entropy or seq2seq loss.  
- **For generative models & reinforcement learning:** Use adversarial loss or policy gradient-based losses.  

---
## What regularization should we choose?

### Choosing the Right Regularization Technique

| **Scenario** | **Recommended Regularization** | **Why?** |
|-------------|--------------------------------|----------|
| **Linear Regression** (High-dimensional features) | **L1 Regularization (Lasso)** | Shrinks some weights to zero, performing feature selection. |
| **Linear Regression** (Multicollinearity present) | **L2 Regularization (Ridge)** | Reduces overfitting by penalizing large coefficients, keeping all features. |
| **Linear Regression** (Feature selection + Stability) | **Elastic Net (L1 + L2)** | Combines Lasso (L1) for feature selection and Ridge (L2) for stability. |
| **Neural Networks (Deep Learning)** | **Dropout** | Randomly drops neurons during training to reduce co-adaptation and overfitting. |
| **Neural Networks (Weight Constraints)** | **L2 Regularization (Weight Decay)** | Penalizes large weights to improve generalization, commonly used in deep learning. |
| **Tree-Based Models (Decision Trees, Random Forests)** | **Pruning** | Reduces overfitting by removing nodes that do not contribute significantly to predictions. |
| **Tree-Based Models (Gradient Boosting, XGBoost, LightGBM)** | **Early Stopping** | Stops training when validation loss stops improving, preventing overfitting. |
| **Imbalanced Classification** | **Class Weighting** | Adjusts loss function weights to give higher importance to minority classes. |
| **Sparse Data (NLP, Text Classification)** | **L1 Regularization** | Helps eliminate irrelevant features (words) by setting some weights to zero. |
| **Highly Correlated Features** | **L2 Regularization (Ridge)** | Helps reduce variance and stabilize coefficients. |
| **Generative Models (GANs, VAEs)** | **Spectral Normalization / Batch Normalization** | Stabilizes training and prevents mode collapse. |
| **Batch Learning with Noisy Data** | **Batch Normalization** | Reduces internal covariate shift and stabilizes learning. |
| **Sequential Learning (Time-Series, RNNs, LSTMs)** | **Gradient Clipping** | Prevents exploding gradients by capping updates to reasonable values. |

### **💡 How to Choose the Right Regularization?**
- Use **L1 (Lasso)** when you need **feature selection**.  
- Use **L2 (Ridge)** when you need **stability and shrinkage of all weights**.  
- Use **Elastic Net** when you need **both feature selection and stability**.  
- Use **Dropout** in deep learning to prevent **overfitting** in neural networks.  
- Use **Pruning or Early Stopping** for **tree-based models** to limit model complexity.  
- Use **Batch Normalization or Gradient Clipping** for **stable training in deep networks**.  

---
## What activation function should we choose?

| **Activation Function** | **Range**              | **Use Case**                                                                 | **When to Avoid**                                                   |
|-------------------------|------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Sigmoid**              | (0, 1)                 | - Binary classification (output layer)                                       | - Deep networks (vanishing gradients)                               |
| **Tanh**                 | (-1, 1)                | - Hidden layers in shallow networks                                           | - Deep networks (vanishing gradients)                               |
| **ReLU**                 | (0, ∞)                 | - Hidden layers in deep networks                                             | - Dying ReLU problem (neurons output 0)                            |
| **Leaky ReLU**           | (-∞, ∞) (with small negative slope) | - Hidden layers in deep networks (avoid dying ReLU problem)                | - Generally no major drawbacks, but slightly slower than ReLU       |
| **ELU**                  | (-∞, ∞)                | - Deep networks, improves convergence                                        | - Computationally more expensive than ReLU                         |
| **Softmax**              | (0, 1) (sum = 1)       | - Multi-class classification (output layer)                                  | - Not for binary classification or regression tasks                |
| **Swish**                | (-∞, ∞)                | - Suitable for deep networks, can outperform ReLU in some tasks              | - May not always outperform ReLU for all problems                   |
| **GELU**                 | (-∞, ∞)                | - Deep networks, NLP (e.g., Transformer-based architectures like GPT, BERT)  | - Computationally more expensive than ReLU                         |
| **Hard Sigmoid / Hard Swish** | (0, 1) or (-1, 1)  | - Approximation of Sigmoid or Swish for computational efficiency              | - Limited flexibility, not suitable for complex tasks              |


---
## What evaluation metric (offline) should we choose?

| **Task**                | **Evaluation Metric**       | **When to Use**                                                       |
|-------------------------|-----------------------------|----------------------------------------------------------------------|
| **Binary Classification**| **Accuracy**                | - When classes are balanced.                                         |
|                         | **Precision**                | - When false positives are costly or need to be minimized (e.g., spam detection). |
|                         | **Recall (Sensitivity)**    | - When false negatives are costly or need to be minimized (e.g., medical diagnosis). |
|                         | **F1 Score**                | - When there is a need to balance Precision and Recall (especially with imbalanced data). |
|                         | **AUC-ROC Curve**           | - For evaluating the tradeoff between True Positive Rate (Recall) and False Positive Rate (1-Specificity). |
|                         | **Log Loss**                | - For probabilistic models, measures the accuracy of the predicted probabilities. |
| **Multi-Class Classification** | **Accuracy**         | - When classes are balanced.                                         |
|                         | **Precision (Macro/Micro)** | - When each class is equally important, or when focusing on precision across all classes. |
|                         | **Recall (Macro/Micro)**    | - When focusing on recall for each class or minimizing false negatives across all classes. |
|                         | **F1 Score (Macro/Micro)**  | - For balancing Precision and Recall across multiple classes, especially for imbalanced datasets. |
|                         | **Confusion Matrix**        | - To understand the classification performance, especially in imbalanced classes. |
| **Regression**           | **Mean Absolute Error (MAE)** | - When you care equally about all errors, or when outliers should be less important. |
|                         | **Mean Squared Error (MSE)** | - When penalizing large errors more than smaller ones is important. |
|                         | **Root Mean Squared Error (RMSE)** | - When you want to interpret the error in the same units as the output. |
|                         | **R-squared (R²)**          | - When you want to understand how well the model explains the variance in the data. |
|                         | **Adjusted R-squared**     | - For comparing models with different numbers of predictors, penalizing unnecessary predictors. |
|                         | **Explained Variance Score** | - When you want to understand the proportion of the variance explained by the model. |
| **Ranking / Recommendation Systems** | **Mean Reciprocal Rank (MRR)** | - When evaluating ranked lists and you care about the position of the first relevant item. |
|                         | **Normalized Discounted Cumulative Gain (NDCG)** | - For evaluating ranked relevance (higher ranks matter more). |
|                         | **Precision at K**          | - To measure how many relevant items appear in the top `K` recommendations. |
| **Anomaly Detection**   | **Precision**                | - When you want to measure the proportion of correctly identified anomalies (positive predictive value). |
|                         | **Recall**                   | - When you want to measure the proportion of all anomalies correctly identified (true positive rate). |
|                         | **F1 Score**                 | - When you want to balance precision and recall, especially in imbalanced anomaly detection tasks. |
| **Clustering**          | **Silhouette Score**         | - To measure the quality of clusters; a higher score indicates well-separated and well-defined clusters. |
|                         | **Davies-Bouldin Index**     | - To measure the average similarity ratio of each cluster with its most similar cluster (lower is better). |
|                         | **Adjusted Rand Index (ARI)** | - For comparing clustering results with ground truth, especially with different cluster sizes. |
| **Time Series Forecasting** | **Mean Absolute Percentage Error (MAPE)** | - When you need to measure prediction accuracy in percentage terms, especially when scale independence is important. |
|                         | **Mean Absolute Scaled Error (MASE)** | - When comparing forecasts to a naive baseline, especially when the data has different scales. |
|                         | **Root Mean Squared Logarithmic Error (RMSLE)** | - When the data has a skewed distribution and you want to reduce the influence of large values. |


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
