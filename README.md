# Power-Outage-Economic-Analysis
DSC 80 Final Project by Karthik Sankaran

Skip to the content.

# ⚡ Analyzing Power Outages and Economic Resilience

Project for DSC 80 at UCSD

---

### View on GitHub
[Link to your GitHub repository]

### Analyzing Power Outages and Economic Output
Project for DSC 80 at UCSD
By [Your Name]

## Introduction

In this project, I examined a data set of **major power outages in the U.S. from January 2000 to July 2016**. These outages, defined by the Department of Energy, impacted at least 50,000 customers or caused substantial demand loss. The data, sourced from Purdue University, includes detailed information on the outages, state demographics, climate, and **economic characteristics**.

My core research question was: **Is there a relationship between a state's economic output (GSP) and its susceptibility to long-duration power outages?** The project culminates in a model predicting the risk of a "Long Outage" and an analysis of model fairness with respect to state economic output.

The original raw DataFrame contained 1534 rows and 57 columns. The primary columns used for analysis are listed below:

| Column | Role in Analysis |
| :--- | :--- |
| `OUTAGE.DURATION` | Used to define the response variable (`LONG_OUTAGE`). |
| `U.S._STATE` | Geographical and categorical feature. |
| `CAUSE.CATEGORY` | Crucial feature used in the Final Model. |
| `TOTAL.REALGSP` | Used for grouping in Hypothesis and Fairness tests. |
| `PC.REALGSP.STATE` | Per capita economic feature used in the Baseline and Final Models. |
| `CUSTOMERS.AFFECTED` | Used to engineer the `CUST_PER_GSP` feature. |

---

## Data Cleaning and Exploratory Data Analysis (EDA)

### Cleaning

Key cleaning steps included:
1.  Handling $0$ values in `OUTAGE.DURATION`, `CUSTOMERS.AFFECTED`, and `DEMAND.LOSS.MW` by replacing them with **`np.nan`**, as a major outage cannot have zero impact.
2.  Creating the response variable: **`LONG_OUTAGE`** (1 if `OUTAGE.DURATION` > median duration, 0 otherwise).
3.  Imputing missing values in economic columns (`TOTAL.REALGSP`, `CUSTOMERS.AFFECTED`) using the **median** before feature engineering.

### Exploratory Data Analysis

* **Initial Aggregation (GSP vs. Duration):** Initial grouping of states by GSP quartile revealed that states in the **Lower GSP quartiles had longer median outage durations** compared to those in the Higher GSP quartiles. This observation formed the basis for the formal Hypothesis Test.

---

## Hypothesis Testing

**Question:** Is the outage duration greater on average for low-GSP states compared to high-GSP states?

| Component | Result | Conclusion |
| :--- | :--- | :--- |
| **Test Statistic** | Difference in Means (Low GSP - High GSP) | $1033.38$ minutes |
| **P-value** | $0.002$ | **Reject $H_0$** (Statistically Significant) |

* **Conclusion:** With a $p$-value well below the $\alpha=0.05$ threshold, we **reject the null hypothesis**. There is **strong statistical evidence** that states with lower economic output experience longer average power outages.

---

## Final Model Development

### Framing the Prediction Problem

* **Prediction Task:** **Binary Classification**—predicting **`LONG_OUTAGE`** (outage > 620 minutes) vs. Short Outage (outage $\leq 620$ minutes).
* **Evaluation Metric:** **Accuracy**. Chosen because the response variable was engineered to be roughly balanced (50/50).

### Baseline Model (Logistic Regression)

| Component | Detail | Performance |
| :--- | :--- | :--- |
| **Model** | `LogisticRegression` | **Accuracy: $0.597$** |
| **Features** | `PC.REALGSP.STATE`, `U.S._STATE` | |

### Final Model (Random Forest)

To significantly improve prediction accuracy, we moved to an ensemble method and introduced new, highly predictive features. 

[Image of Decision Tree structure]


| Component | Detail | Performance |
| :--- | :--- | :--- |
| **Model** | `RandomForestClassifier` (optimized) | **Test Accuracy: $0.791$** |
| **Tuning** | `GridSearchCV` used for hyperparameter selection. | |
| **Features Added** | **1. `CAUSE.CATEGORY`**: Direct predictor of repair complexity. **2. `CUST_PER_GSP`**: (Customers Affected / Total GSP) - measures outage scale relative to state economic capacity. | |
| **Best Hyperparameters** | `max_depth`: 5, `n_estimators`: 100 | |

The Final Model achieved a test accuracy of **$0.791$**, representing a significant improvement over the $0.597$ Baseline, validating the new features and the non-linear model choice.

---

## Fairness Analysis

We assessed whether the Final Model's accuracy was fair with respect to economic output using a permutation test.

### 1. Hypotheses

* **Groups:** Group X (Lower GSP States, below median) and Group Y (Higher GSP States, at or above median).
* **Null Hypothesis ($H_0$):** The model is fair. The true accuracy for Lower GSP states ($\mu_X$) is the same as for Higher GSP states ($\mu_Y$).
* **Alternative Hypothesis ($H_A$):** The model is unfair. The true accuracy for the two groups is different ($\mu_X \neq \mu_Y$).

### 2. Permutation Test Results

| Component | Value |
| :--- | :--- |
| **Accuracy (Group X - Lower GSP)** | $0.851$ |
| **Accuracy (Group Y - Higher GSP)** | $0.723$ |
| **Observed Difference ($\text{Acc}_X - \text{Acc}_Y$)** | $0.128$ |
| **P-value** | $0.092$ |

### 3. Conclusion

Using a significance level of $\alpha=0.05$, the $p$-value ($0.092$) is **greater than $0.05$**. We **fail to reject the null hypothesis**.

Although the model shows a large observed disparity (it is $12.8\%$ more accurate for Lower GSP states), the permutation test indicates that this difference is **not statistically significant**. We do not have sufficient statistical evidence to conclude that the model is fundamentally unfair with respect to the state's economic output.

***



