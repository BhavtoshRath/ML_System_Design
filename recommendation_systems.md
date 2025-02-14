### Recommendation Techniques: Content-Based vs. Collaborative vs. Hybrid Filtering  

| Feature               | Content-Based Filtering | Collaborative Filtering | Hybrid Filtering |
|-----------------------|------------------------|------------------------|------------------|
| **Approach** | Recommends items similar to what a user has liked before based on item attributes. | Recommends items based on user interactions and preferences of similar users. | Combines both content-based and collaborative filtering approaches. |
| **Data Used** | Item features (e.g., genre, keywords, description). | User-item interactions (ratings, clicks, purchases). | Both item features and user interaction data. |
| **Personalization** | Highly personalized based on the user’s preferences. | Based on community/user trends rather than individual attributes. | More accurate and diverse recommendations. |
| **Cold Start Problem** | Struggles with new users (needs past interactions). | Struggles with new items (needs user interactions). | Mitigates cold start issues by leveraging both user and item data. |
| **Scalability** | Scalable if item features are well-defined. | Can be computationally expensive for large datasets. | More complex but balances scalability and accuracy. |
| **Example** | Netflix recommending movies based on a user’s previously watched genres. | Amazon recommending products based on what similar users bought. | Spotify combining user listening history with trends from similar users to recommend songs. |

### 📌 Key Takeaways  
- **Content-Based Filtering** is best when detailed item attributes are available.  
- **Collaborative Filtering** is useful when there is rich user interaction data.  
- **Hybrid Filtering** provides the most accurate recommendations by combining both approaches.  

---

### Matrix Factorization 
Matrix Factorization is a powerful technique in recommender systems that helps predict user preferences by breaking down 
a big user-item ratings matrix into smaller user features matrix (U) and item features matrix (V). By multiplying U and V^T, 
we approximate missing values in R, allowing us to predict user preferences for unrated items.

```math
R \approx U \times V^T
```

U and V have random values initially. It keeps adjusting them by minimizing the error between actual user-item matrix and 
predicted ratings (R). It then uses optimization techniques like Gradient Descent or Alternating Least Squares (ALS).

**pros**: Very fast, thus scalable
**cons**: Cold start problem


---

### Two-Tower Neural Network

#### **Two Separate Towers**  
- One tower processes **user features** (e.g., age, location, interactions).  
- The other tower processes **item features** (e.g., genre, metadata).  

#### **Feature Encoding**  
- Each tower converts inputs into **low-dimensional embeddings** that capture latent factors.  

#### **Similarity Calculation**  
- The **user and item embeddings** are combined using a similarity function (e.g., **dot product, cosine similarity**).  
- The final score represents the **relevance of the item to the user**.  

---

### Matrix Factorization vs. Two-Tower Neural Network  


| **Aspect**              | **Matrix Factorization**                                                      | **Two-Tower Neural Network** |
|------------------------|-------------------------------------------------------------------------------|------------------------------|
| **Approach**          | Factorizes the user-item interaction matrix into two lower-dimensional matrices. | Uses two separate neural networks to learn embeddings for users and items independently. |
| **Input Data**        | Requires an interaction matrix (explicit ratings, clicks, etc.).              | Can use structured and unstructured data (e.g., metadata, images, text). |
| **Feature Learning**  | Learns latent factors through decomposition.                                  | Learns user-item embeddings via deep learning. |
| **Cold Start Handling** | Struggles with new users/items due to lack of prior interactions.             | Handles new users/items better by leveraging metadata and features. |
| **Scalability**       | Computationally efficient but limited in capturing complex relationships.     | More flexible but computationally expensive, requiring GPU acceleration. |
| **Similarity Calculation** | Uses dot product between latent factors of users and items.                   | Uses a similarity function (e.g., dot product, cosine similarity) on embeddings. |
| **Flexibility**       | Limited to structured user-item interaction data.                             | Supports diverse data sources (text, images, metadata, etc.). |
| **Accuracy in Complex Cases** | Works well for linear relationships but struggles with non-linear patterns.   | Captures complex, non-linear relationships using deep networks. |
| **Use Cases**        | Collaborative filtering in movie, music, and e-commerce recommendations.      | Search ranking, ad personalization, product recommendations, and large-scale matching tasks. |
