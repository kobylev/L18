# 📄 Project Requirements Document (PRD): Binary Classification via Logistic Regression

## 1. 🎯 Goal of the Assignment (Objective)

The primary objective is to implement a **Logistic Regression** algorithm *manually*, utilizing the **Sigmoid function** and the **Gradient Descent** (specifically Gradient *Ascent*) method, to find the optimal $\beta$ coefficients ($\beta_0, \beta_1, \beta_2$) required for accurate **binary classification** of two distinct, well-separated data clusters.

---

## 2. 🏗️ Phase 1: Data Preparation - Synthetic Dataset Creation

The assignment requires generating a synthetic dataset containing two absolutely separable clusters for binary classification.

| Component | Description |
| :--- | :--- |
| **Dimensions** | Two independent variables: $X_1$ and $X_2$. |
| **Bias Term ($\beta_0$)** | An additional constant feature $X_0$ must be included, where its value is always **1** (representing the bias term). |
| **Class 0 (Response = 0)** | A collection of points defined as Class 0. |
| **Class 1 (Response = 1)** | A collection of points defined as Class 1. |
| **Normalization** | **Recommendation:** Normalize the $X_1$ and $X_2$ values to be within the range **[0, 1]**. |
| **Dataset Size** | It is recommended to create a **large number of points** (e.g., 5,000 to 10,000) to ensure effective convergence. |

---

## 3. ⚙️ Phase 2: Gradient Descent Implementation

The implementation must be done **without using pre-built machine learning library functions** (like `scikit-learn`'s Logistic Regression). The core logic should be built using loops (for iterations) and NumPy (for linear algebra).

### 3.1. Initial Settings

* **Initialization of $\beta$ Coefficients**: Start the process with **random values** for the three $\beta$ coefficients: $\beta_0, \beta_1, \beta_2$.
* **Learning Rate ($\alpha$ / Learning Step)**: Define a fixed value for the learning step (e.g., 0.03).
* **Sigmoid Function ($\sigma$)**:
    $$\sigma(Z) = \frac{1}{1 + e^{-Z}}$$
    where $Z$ is the weighted sum (linear combination) of the coefficients and the variables:
    $$Z = \beta_0 X_0 + \beta_1 X_1 + \beta_2 X_2$$

### 3.2. Iterative Process (Gradient Ascent)

The algorithm proceeds iteratively until convergence is achieved (e.g., when the change in the Likelihood or Error is minimal). The goal is to find the **maximum** Log-Likelihood.

In each iteration, the following steps must be performed for **every data sample**:

1.  **Calculate Weighted Sum ($Z$)**: Compute $Z$ using the current $\beta$ values for the sample.
    $$Z_i = \beta_0^{(t)} X_{0,i} + \beta_1^{(t)} X_{1,i} + \beta_2^{(t)} X_{2,i}$$
2.  **Calculate Probability ($\hat{P}$) using Sigmoid**:
    $$\hat{P}_i = \sigma(Z_i)$$
3.  **Calculate the Gradient (Derivative)**: Compute the partial derivative of the Log-Likelihood function with respect to each $\beta_k$:
    $$\frac{\partial L}{\partial \beta_k} = (Y_i - \hat{P}_i) \cdot X_{k,i}$$
4.  **Update $\beta$ Coefficients (Gradient Ascent)**: Update the $\beta$ coefficients using the calculated gradients and the learning step. The update uses addition because the goal is **maximizing** the Log-Likelihood (Gradient Ascent).
    $$\beta_k^{(t+1)} = \beta_k^{(t)} + \alpha \cdot \sum_{i=1}^{N} \frac{\partial L_i}{\partial \beta_k}$$

---

## 4. 📈 Phase 3: Deliverables and Submission Requirements

The final submission must include three main components:

### 4.1. Results Table (Tabular Data)

Present a table summarizing the results of the final classification using the optimized $\beta$ coefficients.

| Column | Description |
| :--- | :--- |
| **$X_1$** | The $X_1$ variable of the sample. |
| **$X_2$** | The $X_2$ variable of the sample. |
| **$Y$ (Ground Truth)** | The true response (0 or 1). |
| **$\text{Sigmoid}(\hat{P})$ (Prediction)** | The probability predicted by the Sigmoid model. |
| **Error Squared ($\text{Error}^2$)** | The square of the momentary difference: $(Y - \hat{P})^2$. |
| **Average Error (MSE)** | **In a separate row at the bottom:** Calculate the Mean Squared Error (MSE) over the entire dataset. |

### 4.2. Classification Plot

Present a graph visualizing the data points and the classification results.

* **Data Points**: Each point on the graph represents a sample based on its $X_1$ and $X_2$ values.
* **True vs. Predicted Classification**:
    * Use a distinct **shape** (e.g., 'X' for true, circle for predicted) to denote the **Ground Truth** ($Y$) and **Prediction**.
    * Use **color** (e.g., Red and Blue) to denote the **Predicted Class** ($P>0.5$ is Class 1, $P\leq 0.5$ is Class 0).
* **Decision Boundary**: Plot the **Sigmoid decision boundary** (the line where $\sigma(Z)=0.5$) that separates the two predicted classes.

### 4.3. Convergence Plot

Present two graphs showing the progress of the learning process across iterations.

| Graph | X-Axis | Y-Axis | Expected Trend |
| :--- | :--- | :--- | :--- |
| **1. Log-Likelihood Progress** | Iteration Number | Log-Likelihood Value | **Increasing** (since we are maximizing the likelihood). |
| **2. Error Progress** | Iteration Number | Average Error Value (MSE) | **Decreasing**. |

---

## 5. 💡 Advanced and Optional Tasks

* **Testing on Unseen Data (Generalization)**:
    1.  Create a **Test Set** of new data points that were **not** used during the training process.
    2.  Pass these unseen points through the Sigmoid function using the **final, optimized $\beta$ coefficients**.
    3.  Demonstrate the model's ability to classify these new points accurately ("From the 'is,' it reveals the 'is not'").
* **Hyperparameter Analysis**: Experiment with different starting $\beta$ values and learning rates ($\alpha$) to analyze their impact on the speed and stability of the convergence.