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

The cleaning phase focused on transforming and preparing the data for analysis. The most critical step involved handling implausible values: zero values in the severity metrics—`OUTAGE.DURATION`, `CUSTOMERS.AFFECTED`, and `DEMAND.LOSS.MW`—were replaced with `np.nan`, as a major outage cannot logically have zero impact. Next, the binary response variable, `LONG_OUTAGE`, was created, set to 1 if the `OUTAGE.DURATION` was greater than the overall median duration, and 0 otherwise. Finally, to ensure features were complete for subsequent engineering and modeling, missing values in the economic columns used for prediction (`TOTAL.REALGSP` and `CUSTOMERS.AFFECTED`) were imputed using the median of each respective column.

### Exploratory Data Analysis

The EDA phase focused on generating initial insights. The primary exploratory action was an Initial Aggregation where states were explicitly divided into four GSP quartiles (`Low`, `Medium-Low`, `Medium-High`, `High`) based on the distribution of their `PC.REALGSP.STATE`. By grouping the data using these bins and calculating the mean and median `OUTAGE.DURATION` for each, a significant observation was made: states in the Lower GSP quartiles had longer median outage durations compared to those in the Higher GSP quartiles. This clear pattern showing that a state's economic output correlates with the severity of its power outages directly supported the decision to formulate and run the subsequent formal Hypothesis Test.

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

The first graph depicts the distribution of outage durations across all recorded outages. From this graph, it is evident that most outages last for little; that is, the overwhelming majority (pre-imputation and missingness analysis) last for less than a few thousand minutes, with only a small number extending into the extreme range. The scatter plot represents outage durations across multiple states. The graph shows that most states are clustered around the ~50k REAL GSP per capita mark on the x-axis, since many states have similar economic output, but outage durations themselves vary widely. This plot does not reflect a unimodal distribution; rather, it shows that outage duration does not clearly increase or decrease with state economic output.

| `PC.REALGSP.STATE` | count | mean | median |
|:-------------------|--------:|--------:|---------:|
| Low                | 174 | 3013.05 | 1066.5 |
| Medium-Low         | 173 | 2867.64 | 960 |
| Medium-High        | 174 | 2140.81 | 709 |
| High               | 172 | 1670.82 | 202.5 |

States in the lower economic output categories tend to experience longer average outage durations compared to states in higher economic categories. This suggests a possible relationship between lower GSP and weaker grid reliability. Another point of curiosity is the fact that the differences in means across categories are much larger than the differences in medians. This indicates that the distributions are heavily right-skewed and that extreme long duration outages (not missing or zero values) are likely inflating the means. We will explore this further in the missingness section.

---

## Assessment of Missingness 

<iframe
  src="assets/missingness_count_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### NMAR Analysis

The column `OUTAGE.RESTORATION.TIME` is hypothesized to be **Not Missing At Random (NMAR)**. The missingness is likely dependent on the value itself: long or complex outages that require tedious documentation procedures are harder for utility companies to record accurately. It is plausible that the reporting entities omit the restoration time altogether due to the complexity of the event. To investigate if this column is MAR instead of NMAR, one could collect report logs or utility company documentation protocols that could explain the missingness by factors other than the outage time itself.

### Missingness Dependency: Investigating MAR

We conducted two permutation tests to determine if the missingness of the `RES.PRICE` column is dependent on other observed variables.

#### Test 1: Dependency on `CUSTOMERS.AFFECTED`

<iframe
  src="assets/perm_customers.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

In the first test, we examined whether the missingness of `RES.PRICE` depended on `CUSTOMERS.AFFECTED`. The observed difference was -94,819, indicating that outages with missing price data affected fewer customers on average. The resulting p-value was 0.329, which is greater than 0.05. Therefore, we fail to reject the null hypothesis, and the missingness of `RES.PRICE` is not statistically associated with the number of customers affected by the outage.

#### Test 2: Dependency on `YEAR`

<iframe
  src="assets/perm_year.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The second test examined whether the missingness of `RES.PRICE` depended on the `YEAR` of the outage. The observed difference was 6.907 years, indicating that outages with missing price data occurred approximately 6.9 years later on average. The p-value was 0.000, which is statistically significant. Therefore, we reject the null hypothesis. This finding suggests that `RES.PRICE` is Missing At Random (MAR), as the likelihood of price data being missing depends on the observed variable `YEAR`. The accompanying visualization confirms this, showing that the percentage of missing `RES.PRICE` data increases sharply in later years of the dataset (post-2010).

---

## Hypothesis Testing

The key question was whether outage duration is greater on average for low-GSP states compared to high-GSP states.

<iframe
  src="assets/perm_gsp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed difference in means between low and high GSP states was 1033.38 minutes, and the p-value was 0.002. This p-value is well below the 0.05 threshold, allowing us to reject the null hypothesis. Therefore, there is strong statistical evidence that states with lower economic output experience longer average power outages.

---

## Final Model Development

### Framing the Prediction Problem

The prediction task was framed as a binary classification problem, predicting `LONG_OUTAGE` (outage > 620 minutes) versus short outages (≤ 620 minutes). Accuracy was chosen as the evaluation metric because the response variable was engineered to be roughly balanced (50/50).

### Baseline Model (Logistic Regression)

The baseline model used `LogisticRegression` with features `PC.REALGSP.STATE` and `U.S._STATE`. This model achieved an accuracy of 0.597, providing a benchmark for subsequent models.

### Final Model (Random Forest)

To significantly improve prediction accuracy, we implemented a `RandomForestClassifier` and introduced two highly predictive features. The first feature, `CAUSE.CATEGORY`, serves as a direct predictor of repair complexity. The second feature, `CUST_PER_GSP` (customers affected divided by total GSP), measures the scale of the outage relative to the state's economic capacity. GridSearchCV was used for hyperparameter optimization, and the best hyperparameters found were `max_depth = 5` and `n_estimators = 100`. The final model achieved a test accuracy of 0.791, representing a significant improvement over the baseline. 

[Image of Decision Tree structure]

---

## Fairness Analysis

We also assessed whether the final model's accuracy was fair with respect to economic output using a permutation test. Group X included lower GSP states (below the median), and Group Y included higher GSP states (at or above the median). The null hypothesis stated that the model is fair, meaning the true accuracy for lower GSP states is the same as for higher GSP states. The alternative hypothesis stated that the model is unfair, with different true accuracies for the two groups.

The permutation test results showed that Group X had an accuracy of 0.851, while Group Y had an accuracy of 0.723. The observed difference (AccX - AccY) was 0.128, and the p-value was 0.092. Since this p-value is greater than the 0.05 significance level, we fail to reject the null hypothesis. Although the model shows a 12.8% higher accuracy for lower GSP states, the difference is not statistically significant. Therefore, there is insufficient evidence to conclude that the model is fundamentally unfair with respect to the state's economic output.
