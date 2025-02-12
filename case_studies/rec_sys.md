# 📌 Recommender System - Case Study

## **1️⃣ Problem Statement** 🎯  
-  'business objective': Increase user engagement, revenue, or any other KPI
- Inputs and outputs to the system
- Rec type: personalized, general or context-aware (ex: google maps)
- Data type: Implicit (clicks, views, other guest behaviour) or explicit (ratings, reviews, likes/dislikes)
- Rec sys challenges: Cold start, Scalability, Real-time capability

---

## **2️⃣ System Architecture** 🏗️

### **🔹 1. Data Ingestion & Storage**  
Collects and stores user interactions, metadata, and contextual signals.  

| **Component**                                                                  | **Technology**                    | **Purpose** |
|--------------------------------------------------------------------------------|-----------------------------------|------------|
| **Event Streaming** (real-time processing of continuous data stream at scale)  | Kafka (Apache), Kinesis (AWS)     | Capture real-time interactions (clicks, views, watch duration). |
| **Batch Storage** (Large-scale data storage for batch procesing and analytics) | S3, HDFS, BigQuery                | Store historical viewing data for model training. |
| **Real-time DB** (Rapid read/write operations at scale/ in real-time)          | DynamoDB(AWS), Cassandra (Apache) | Store active user sessions for real-time recommendations. |
| **Feature Store** (ML features management tool)                                | Redis, Feast                      | Store processed user/item embeddings. |

---

### **🔹 2. Model Training & Feature Engineering**

| **Model Type** | **Algorithm** | **Use Case** |
|--------------|-------------|------------|
| **Collaborative Filtering** | Matrix Factorization (ALS, SVD) | Find similar users & items based on interactions. |
| **Content-Based** | TF-IDF, Word2Vec | Recommend based on movie descriptions, cast, and genre. |
| **Deep Learning** | Two-Tower Network, Transformers | Capture non-linear relationships between users & content. |
| **Contextual Bandits** | Multi-Armed Bandit | Adapt recommendations based on real-time feedback. |

---

### **🔹 3. Recommendation Serving**  

| **Component** | **Technology**                            | **Purpose**                                                                                                                      |
|--------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Model Serving** | TensorFlow Serving, TorchServe            | Deploy trained models for inference.                                                                                             |
| **Recommendation API** | FastAPI (uses JSON), gRPC (uses protobuf) | REST API (allows applications to communicate with each other over HTTP using REST principles). Serves recommendations to the UI. |
| **Retrieval & Ranking** | FAISS, Annoy, ScaNN                       | Fast nearest neighbor search for recommendations.                                                                                |
| **Caching** | Redis, Memcached                          | Cache top-N recommendations for low latency.                                                                                     |

---

## **3️⃣ Scaling Considerations** 🌍   

| **Scaling Factor** | **Solution** |
|-------------------|-------------|
| **Massive User Base** | Distributed data storage using **Cassandra** or **BigQuery** ensures data is partitioned across multiple nodes to handle high traffic loads efficiently. This allows the system to scale horizontally as more users join. |
| **Real-Time Personalization** | **Kafka** and **Flink** process real-time user interactions, ensuring that new engagement signals (such as watch history or search queries) update recommendations almost instantly. This prevents stale recommendations and enhances user satisfaction. |
| **High Throughput Requests** | A combination of **load balancing** (NGINX, Envoy) and **caching mechanisms** (Redis, CDN) ensures that frequent recommendations are served quickly, reducing backend computational load and improving latency. |
| **Frequent Model Updates** | Automated retraining pipelines using **Airflow** and **Kubeflow** allow models to continuously evolve by integrating fresh user interaction data, ensuring relevance and improving recommendation accuracy over time. |


---

## **4️⃣ Evaluation & Metrics** 📊  

### **🔹 Offline Metrics** (Model Training)  
- **Precision/Recall** → Measures recommendation `accuracy`
- **NDCG (Normalized Discounted Cumulative Gain)** → Evaluates `ranking quality`  
- **MAP (Mean Average Precision)** → Assesses `ranking relevance`

### **🔹 Online Metrics** (User Engagement)  
- **Click-Through Rate (CTR)** → Measures how often users click on recommendations.  
- **Watch Time** (movies; songs), **Attributable Demand** (e-commerce)
- **A/B Testing** → Compares new vs. old recommendation models in production.  
