
# Data Scaling: 

### Normalization vs. Standardization vs. Log Scaling

| **Feature**      | **Normalization** | **Standardization** | **Log Scaling** |
|-----------------|------------------|--------------------|----------------|
| **Definition**   | Scales data within a fixed range (e.g., [0,1] or [-1,1]). | Transforms data to have a mean of 0 and a standard deviation of 1. | Applies a logarithmic transformation to compress large values. |
| **Formula**      | `X' = (X - X_min) / (X_max - X_min)` | `X' = (X - μ) / σ` | `X' = log(X + ε)`, where `ε` is a small constant to avoid log(0). |
| **Purpose**      | Ensures all values fall within a specific range, useful for models sensitive to scale. | Centers and scales data for better distribution, often for normally distributed inputs. | Reduces skewness and handles data with large variations. |
| **Effect on Data** | Compresses values into a smaller range; sensitive to outliers. | Adjusts distribution to standard normal; less affected by outliers. | Reduces the impact of extreme values and makes distributions more symmetric. |
| **Use Cases**    | - When features have different scales.  <br> - When using distance-based algorithms (e.g., KNN, Neural Networks). <br> - When data needs to be in a fixed range (e.g., image processing). | - When data follows (or approximately follows) a normal distribution. <br> - When using algorithms assuming normally distributed inputs (e.g., Linear Regression, PCA). <br> - When feature importance interpretation is required. | - When data is highly skewed. <br> - When feature values vary over multiple orders of magnitude. <br> - Used in financial, scientific, and biological data (e.g., income, population sizes, gene expression). |
| **Example**      | Converting ages (e.g., 18-65) into a 0-1 range for neural networks. | Standardizing test scores (e.g., SAT scores) for better comparability. | Converting income data (e.g., from $1K to $1M) to reduce skewness. |

---

### When to Use Which?
- **Use Normalization** when the dataset has varying scales and when algorithms like neural networks, KNN, or distance-based methods require values within a specific range.
- **Use Standardization** when data follows a normal distribution or when using algorithms that assume normally distributed inputs, such as linear regression and PCA.
- **Use Log Scaling** when data is highly skewed or spans several orders of magnitude, such as financial transactions or population data.


---