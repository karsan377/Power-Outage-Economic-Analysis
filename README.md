# Power-Outage-Economic-Analysis
DSC 80 Final Project by Karthik Sankaran

## Introduction

This project revolves around power outage data provided by Purdue University from 2000 to 2016. Each row represents a distinct outage for a given state. For each outage information about cause, location, timing, and economic and demographic context is also available. Given this information, the question I am considering is:

What relationship do state economies share with power outages in the U.S.? Specifically, do states with lower economic output (lower real GSP column) experience more frequent or longer power outages compared to states with higher economic output?

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

The cleaning phase focused on transforming and preparing the data for analysis. The most critical step involved handling implausible values: zero values in the severity metrics—**OUTAGE.DURATION**, **CUSTOMERS.AFFECTED**, and **DEMAND.LOSS.MW**—were replaced with **np.nan**, as a major outage cannot logically have zero impact. Next, the binary response variable, **LONG\_OUTAGE**, was created, set to 1 if the OUTAGE.DURATION was greater than the overall median duration, and 0 otherwise. Finally, to ensure features were complete for subsequent engineering and modeling, missing values in the economic columns used for prediction (**TOTAL.REALGSP** and **CUSTOMERS.AFFECTED**) were imputed using the median of each respective column.

### Exploratory Data Analysis

The EDA phase focused on generating initial insights. The primary exploratory action was an Initial Aggregation where states were explicitly divided into four GSP quartiles (Low, Medium-Low, Medium-High, High) based on the distribution of their PC.REALGSP.STATE. By grouping the data using these bins and calculating the mean and median OUTAGE.DURATION for each, a significant observation was made: states in the Lower GSP quartiles had longer median outage durations compared to those in the Higher GSP quartiles. This clear pattern showing that a state's economic output correlates with the severity of its power outages directly supported the decision to formulate and run the subsequent formal Hypothesis Test.

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

<p style="font-weight: normal;">
The first graph depicts the distribution of outage durations across all recorded outages. From this graph, it is evident that most outages last for little; that is, the overwhelming majority (pre-imputation and missingness analysis) last for less than a few thousand minutes, with only a small number extending into the extreme range. The scatter plot represents outage durations across multiple states. The graph shows that most states are clustered around the ~50k REAL GSP per capita mark on the x-axis, since many states have similar economic output, but outage durations themselves vary widely. This plot does not reflect a unimodal distribution; rather, it shows that outage duration does not clearly increase or decrease with state economic output.
</p>

| PC.REALGSP.STATE |   count |    mean |   median |
|:-------------------|--------:|--------:|---------:|
| Low                |     174 | 3013.05 |   1066.5 |
| Medium-Low         |     173 | 2867.64 |    960   |
| Medium-High        |     174 | 2140.81 |    709   |
| High               |     172 | 1670.82 |    202.5 |

<p style="font-weight: normal;">
States in the lower economic output categories tend to experience longer average outage durations compared to states in higher economic categories. This suggests a possible relationship between lower GSP and weaker grid reliability. Another point of curiosity is the fact that the differences in means across categories are much larger than the differences in medians. This indicates that the distributions are heavily right-skewed and that extreme long duration outages (not missing or zero values) are likely inflating the means. We will explore this further in the missingness section.
</p>

---

## Assessment of Missingness 🕵️‍♀️

### NMAR Analysis

The column **`OUTAGE.RESTORATION.TIME`** is hypothesized to be **Not Missing At Random (NMAR)**. The missingness is likely dependent on the value itself: long or complex outages that require tedious documentation procedures are harder for utility companies to record accurately. It is plausible that the reporting entities omit the restoration time altogether due to the complexity of the event.

* **Data to Test NMAR:** To investigate if this column is MAR instead of NMAR, one could collect **report logs** or **utility company documentation protocols** that could explain the missingness by factors other than the outage time itself.

### Missingness Dependency: Investigating MAR

We conducted two permutation tests to determine if the missingness of the **`RES.PRICE`** column is dependent on other observed variables.

#### Test 1: Dependency on `CUSTOMERS.AFFECTED`

| Component | Result |
| :--- | :--- |
| **Observed Difference** | $-94,819$ (Outages with missing price data affected fewer customers on average). |
| **P-value** | $0.329$ |

**Conclusion:** Since $p=0.329$ is greater than $0.05$, we **fail to reject the null hypothesis**. The missingness of `RES.PRICE` is **not statistically associated** with the number of customers affected by the outage.

#### Test 2: Dependency on `YEAR`

| Component | Result |
| :--- | :--- |
| **Observed Difference** | $6.907$ years (Outages with missing price data occurred $\sim 6.9$ years later on average). |
| **P-value** | $0.000$ |

**Conclusion:** With $p=0.000$, we **reject the null hypothesis**. There is a statistically significant difference between `YEAR` and `RES.PRICE` missingness. This finding suggests that `RES.PRICE` is **Missing At Random (MAR)**, where the likelihood of price data being missing depends on the observed variable **`YEAR`**.

**Visualization of Dependency:**

<iframe
  src="assets/res_price_missing_by_year.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot confirms the dependency, showing that the percentage of missing `RES.PRICE` data increases sharply in the later years of the dataset (post-2010).

---

## Hypothesis Testing

**Question:** Is the outage duration greater on average for low-GSP states compared to high-GSP states?

| Component | Result | Conclusion |
| :--- | :--- | :--- |
| **Test Statistic** | Difference in Means (Low GSP - High GSP) | $1033.38$ minutes |
| **P-value** | $0.002$ | **Reject $H_0$** (Statistically Significant) |

* **Conclusion:** With a **p-value well below the 0.05 threshold**, we **reject the null hypothesis**. There is **strong statistical evidence** that states with lower economic output experience longer average power outages.

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
| **Features Added** | **1. `CAUSE.CATEGORY`**: Direct predictor of repair complexity. **2. `CUST_PER_GSP`**: (Customers Affected / Total GSP) measures outage scale relative to state economic capacity. | |
| **Best Hyperparameters** | `max_depth`: 5, `n_estimators`: 100 | |

The Final Model achieved a test accuracy of **$0.791$**, representing a significant improvement over the $0.597$ Baseline, validating the new features and the non-linear model choice.

---

## Fairness Analysis

We assessed whether the Final Model's accuracy was fair with respect to economic output using a permutation test.

### 1. Hypotheses

* **Groups:** Group X (Lower GSP States, below median) and Group Y (Higher GSP States, at or above median).
* **Null Hypothesis (H0):** The model is fair. The true accuracy for Lower GSP states (μX) is the same as for Higher GSP states (μY).
* **Alternative Hypothesis (HA):** The model is unfair. The true accuracy for the two groups is different (μX ≠ μY).

### 2. Permutation Test Results

| Component | Value |
| :--- | :--- |
| **Accuracy (Group X - Lower GSP)** | $0.851$ |
| **Accuracy (Group Y - Higher GSP)** | $0.723$ |
| **Observed Difference (AccX - AccY)** | $0.128$ |
| **P-value** | $0.092$ |

### 3. Conclusion

Using a significance level of **0.05**, the **p-value (0.092) is greater than 0.05**. We **fail to reject the null hypothesis**.

Although the model shows a large observed disparity (it is $12.8\%$ more accurate for Lower GSP states), the permutation test indicates that this difference is **not statistically significant**. We do not have sufficient statistical evidence to conclude that the model is fundamentally unfair with respect to the state's economic output.

***