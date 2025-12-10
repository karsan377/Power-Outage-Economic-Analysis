# Power Outage Economic Analysis
By Karthik Sankaran

## Introduction

This project revolves around power outage data provided by Purdue University from 2000 to 2016. Each row represents a distinct outage for a given state. For each outage, information about cause, location, timing, and economic and demographic context is available. Given this information, the question I am considering is:

What relationship do state economies share with power outages in the U.S.? Specifically, do states with lower economic output experience more frequent or longer power outages compared to states with higher economic output?

This question matters because economic conditions may influence grid investment, infrastructure quality, and resilience.

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

The cleaning phase focused on transforming and preparing the data for analysis. The most critical step involved handling implausible values: zero values in the severity metrics—`OUTAGE.DURATION`, `CUSTOMERS.AFFECTED`, and `DEMAND.LOSS.MW` were replaced with `np.nan`, as a major outage cannot logically have zero impact. Next, the binary response variable, `LONG_OUTAGE`, was created, set to 1 if the `OUTAGE.DURATION` was greater than the overall median duration, and 0 otherwise. Finally, to ensure features were complete for subsequent engineering and modeling, missing values in the economic columns used for prediction (`TOTAL.REALGSP` and `CUSTOMERS.AFFECTED`) were imputed using the median of each respective column.

### Exploratory Data Analysis

The EDA phase focused on generating initial insights. The primary exploratory action was a clustering where states were explicitly divided into four GSP quartiles (`Low`, `Medium-Low`, `Medium-High`, `High`) based on the distribution of their `PC.REALGSP.STATE`. 

| `PC.REALGSP.STATE` | count | mean | median |
|:-------------------|--------:|--------:|---------:|
| Low                | 174 | 3013.05 | 1066.5 |
| Medium-Low         | 173 | 2867.64 | 960 |
| Medium-High        | 174 | 2140.81 | 709 |
| High               | 172 | 1670.82 | 202.5 |

By grouping the data using these bins and calculating the mean and median `OUTAGE.DURATION` for each, a significant observation was made: states in the Lower GSP quartiles had longer median outage durations compared to those in the Higher GSP quartiles. This clear pattern showing that a state's economic output correlates with the severity of its power outages directly supported the decision to formulate and run the upcoming formal hypothesis test. Another point of curiosity is the fact that the differences in means across categories are much larger than the differences in medians. This indicates that the distributions are heavily right-skewed and that extreme long duration outages (not missing or zero values) are likely inflating the means. We will explore this further in the missingness section.

<iframe
  src="assets/outage_duration_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/gsp_vs_duration_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The first graph depicts the distribution of outage durations across all recorded outages. From this graph, it is evident that most outages last for little; that is, the overwhelming majority (pre-imputation and missingness analysis) last for less than 10 thousand minutes, with only a small number extending into the extreme range. The scatter plot represents outage durations across multiple states. The graph shows that most states are clustered around the ~50k REAL GSP per capita mark on the x-axis, since many states have similar economic output, but outage durations themselves vary widely. This plot presents a unimodal distribution; rather, it shows that outage duration does not clearly increase or decrease with state economic output.

---

## Assessment of Missingness 

<iframe
  src="assets/missingness_count_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### NMAR Analysis

The column `OUTAGE.RESTORATION.TIME` is hypothesized to be **Not Missing At Random (NMAR)**. The missingness is likely dependent on the value itself because long or complex outages that require tedious documentation procedures are harder for utility companies to record accurately. It is possible that the reporting entities omit the restoration time due to the complexity of the event.

* **Data to Test NMAR:** To investigate if this column is MAR instead of NMAR, one could collect **report logs** or **utility company documentation protocols** that could explain the missingness by factors other than the outage time itself.

### Missingness Dependency: Investigating MAR

I conducted two permutation tests to determine if the missingness of the `RES.PRICE` column is dependent on other observed variables.

#### Test 1: Dependency on `CUSTOMERS.AFFECTED`

<iframe
  src="assets/perm_customers.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

| Component | Result |
| :--- | :--- |
| **Observed Difference** | -94,819 (Outages with missing price data affected fewer customers on average). |
| **P-value** | 0.329 |

**Conclusion:** Since p=0.329 is greater than 0.05, we **fail to reject the null hypothesis**. The missingness of `RES.PRICE` is **not statistically associated** with the number of customers affected by the outage.

#### Test 2: Dependency on `YEAR`

<iframe
  src="assets/perm_year.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

| Component | Result |
| :--- | :--- |
| **Observed Difference** | 6.907 years (Outages with missing price data occurred ~6.9 years later on average). |
| **P-value** | 0.000 |

**Conclusion:** With p=0.000, we **reject the null hypothesis**. There is a statistically significant difference between `YEAR` and `RES.PRICE` missingness. This finding suggests that `RES.PRICE` is **Missing At Random (MAR)**, where the likelihood of price data being missing depends on the observed variable `YEAR`.

**Visualization of Dependency:**

<iframe
  src="assets/res_price_missing_by_year.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot confirms the dependency, showing that the percentage of missing `RES.PRICE` data increases sharply in the later years of the dataset (post-2-15).

---

## Hypothesis Testing

**Question:** Is the outage duration greater on average for low-GSP states compared to high-GSP states?

<iframe
  src="assets/perm_gsp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

| Component | Result | Conclusion |
| :--- | :--- | :--- |
| **Test Statistic** | Difference in Means (Low GSP - High GSP) | 1033.38 minutes |
| **P-value** | 0.002 | **Reject H0** (Statistically Significant) |

With a p-value well below the 0.05 threshold, we **reject the null hypothesis**. There is strong statistical evidence that states with lower economic output experience longer average power outages.

---

## Final Model Development

### Framing the Prediction Problem

* **Prediction Task:** Binary Classification—predicting `LONG_OUTAGE` (outage > 620 minutes) vs. Short Outage (outage ≤ 620 minutes).
* **Evaluation Metric:** Accuracy. Chosen because the response variable was engineered to be roughly balanced (50/50).

### Baseline Model (Logistic Regression)

| Component | Detail | Performance |
| :--- | :--- | :--- |
| **Model** | `LogisticRegression` | Accuracy: 0.597 |
| **Features** | `PC.REALGSP.STATE`, `U.S._STATE` | |

### Final Model (Random Forest)

To significantly improve prediction accuracy, I made a binary classifier predicing whether an outage would be long (greater than 620 minutes) or short (620 minutes or less). This evaluation metric chosen was accuracy because the output classes were engineering to be roughly balanced between long or short outages.

The baseline model, which employed logistic regression, achieved an accuracy of 0.597, indicating that basic features have limited predicting power and were possibly underfitting. To improve performance, I shifted to a Random Forest classifier capable of capturing complex non-linear relationships in the data.


| Component | Detail | Performance |
| :--- | :--- | :--- |
| **Model** | `RandomForestClassifier` (optimized) | Test Accuracy: 0.791 |
| **Tuning** | GridSearchCV used for hyperparameter selection | |
| **Features Added** | 1. `CAUSE.CATEGORY`: Direct predictor of repair complexity. 2. `CUST_PER_GSP`: (Customers Affected / Total GSP) measures outage scale relative to state economic capacity | |
| **Best Hyperparameters** | `max_depth`: 5, `n_estimators`: 100 | |

The final model incorporated two highly informative features. First, CAUSE.CATEGORY represents the type of outage, which is strongly associated with repair complexity and duration. Second, CUST_PER_GSP is an interaction feature calculated as the number of customers affected divided by the state’s total economic output (GSP). This feature captures the scale of the outage relative to the state’s economic capacity, providing a more context. Hyperparameter optimization using GridSearchCV identified the best model as a Random Forest with 100 trees and a maximum depth of 5. On the test set, this final model achieved an accuracy of 0.791, representing a substantial improvement over the baseline as expected.

---

## Fairness Analysis

After developing the final model, I assessed its fairness with respect to economic output using a permutation test. I divided states into two groups: Group X, with lower GSP states below the median, and Group Y, made up of higher GSP states at or above the median.

### 1. Hypotheses

* **Groups:** Group X (Lower GSP States, below median) and Group Y (Higher GSP States, at or above median).  
* **Null Hypothesis (H0):** The model is fair. The true accuracy for Lower GSP states (μX) is the same as for Higher GSP states (μY).  
* **Alternative Hypothesis (HA):** The model is unfair. The true accuracy for the two groups is different (μX ≠ μY).

The null hypothesis was that the the accuracy for lower GSP states is equal to that of higher GSP states. The alternative hypothesis suggested that the model is unfair, with differing accuracies across both groups.

### 2. Permutation Test Results

<iframe
  src="assets/fairness_perm_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

| Component | Value |
| :--- | :--- |
| **Accuracy (Group X - Lower GSP)** | 0.851 |
| **Accuracy (Group Y - Higher GSP)** | 0.723 |
| **Observed Difference (AccX - AccY)** | 0.128 |
| **P-value** | 0.092 |

The test revealed that the model achieved an accuracy of 0.851 for lower GSP states and 0.723 for higher GSP states, resulting in an observed difference of 0.128. The permutation test produced a p-value of 0.092, which is above the conventional significance threshold of 0.05. There, we couldn't reject the null hypothesis. Although the model exhibits a noticibly different measure in accuracy, our permutation test reveals that this difference can be explained by random chance. Therefore, we don't have evidence to conclude that the model is unfair with respect to a state's economic output, suggesting that the model is reasonably balanced.



