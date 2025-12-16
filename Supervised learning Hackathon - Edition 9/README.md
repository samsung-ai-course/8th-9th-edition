# Supervised Learning Hackathon - Edition 9

## Overview

Welcome to the Supervised Learning Hackathon! In this challenge, you will build a machine learning model to predict loan approval status based on various applicant features. Your goal is to create the most accurate model possible using the training data provided.

## Challenge

Predict whether a loan application will be **approved** (1) or **rejected** (0) based on applicant information such as age, income, credit score, loan amount, and more.

**Beyond Accuracy**: This hackathon emphasizes responsible AI practices. You must not only build an accurate model but also:
- Analyze your model for potential biases across demographic groups
- Perform decision threshold analysis to understand precision-recall tradeoffs
- Justify your threshold choice considering both business and ethical implications

## Dataset Files

### Files You Have Access To:

1. **history.csv** (38,250 rows)
   - Training dataset with all features including the target variable `loan_status`
   - Use this file to train and validate your machine learning model
   - Contains 85% of the complete dataset

2. **test.csv** (6,750 rows)
   - Test dataset WITHOUT the target variable `loan_status`
   - Use your trained model to predict loan status for these records
   - Contains 15% of the complete dataset

3. **sample_submission.csv**
   - Template file showing the correct submission format
   - Contains two columns: `id` and `pred`
   - Your submission file must match this format exactly

### Files NOT Available to Students:

- **all_data.csv** - Complete dataset with IDs (for reference only)
- **test_with_labels.csv** - Test dataset with true labels (used for evaluation by teacher)

## Dataset Documentation

The dataset contains 45,000 records and 14 variables, each described below:

| Column | Description | Type |
|--------|-------------|------|
| `person_age` | Age of the person | Float |
| `person_gender` | Gender of the person | Categorical |
| `person_education` | Highest education level | Categorical |
| `person_income` | Annual income | Float |
| `person_emp_exp` | Years of employment experience | Integer |
| `person_home_ownership` | Home ownership status (e.g., rent, own, mortgage) | Categorical |
| `loan_amnt` | Loan amount requested | Float |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_int_rate` | Loan interest rate | Float |
| `loan_percent_income` | Loan amount as a percentage of annual income | Float |
| `cb_person_cred_hist_length` | Length of credit history in years | Float |
| `credit_score` | Credit score of the person | Integer |
| `previous_loan_defaults_on_file` | Indicator of previous loan defaults | Categorical |
| `loan_status` (target variable) | Loan approval status: 1 = approved; 0 = rejected | Integer |

## Instructions

### Step 1: Load and Explore the Data

Start by loading `history.csv` and exploring the dataset.

### Step 2: Build Your Machine Learning Model

1. **Preprocess the data**: Handle missing values, encode categorical variables, scale features if needed
2. **Split your training data**: Create a validation set to evaluate your model
3. **Train your model**: Use any machine learning algorithm (Logistic Regression, Random Forest, XGBoost, Neural Networks, etc.)
4. **Evaluate performance**: Use metrics like accuracy, precision, recall, F1-score, or AUC-ROC
5. **Tune hyperparameters**: Optimize your model for best performance

### Step 3: Make Predictions on Test Set

Load `test.csv` and generate predictions.

### Step 4: Bias Analysis and Decision Threshold Optimization

**REQUIRED**: Before finalizing your predictions, you must analyze your model for bias and optimize the decision threshold.

#### 4.1 Bias Analysis

Analyze whether your model exhibits bias across different demographic groups.

Consider analyzing:
- Performance differences across gender groups
- Impact on different education levels
- Fairness across age groups or home ownership status
- Disparate impact on different income ranges

**Questions to address:**
- Does your model perform differently for different demographic groups?
- Are certain groups more likely to receive false rejections or false approvals?
- What are the potential real-world implications of any bias you observe?

#### 4.2 Decision Threshold Analysis

Instead of using the default 0.5 threshold, analyze the precision-recall tradeoff.

**You must choose a decision threshold and justify your choice:**

Consider the business context:
- **High Recall (Lower Threshold)**: Approve more loans, but risk approving bad loans (more false positives)
  - Use case: Maximize customer acquisition, willing to accept some risk
- **High Precision (Higher Threshold)**: Approve only confident loans, but reject some good applicants (more false negatives)
  - Use case: Risk-averse lending, protect against defaults
- **Balanced Approach**: Find optimal balance based on business costs

**Required Justification:**
In your submission, include a brief report (PDF or markdown) explaining:
1. What bias patterns did you observe in your model?
2. What decision threshold did you choose and why?
3. What is the precision-recall tradeoff at your chosen threshold?
4. Why is this tradeoff appropriate for a loan approval system?
5. What are the potential consequences of your choice for loan applicants?

Example justification:
> "I chose a threshold of 0.45 (slightly favoring recall) because rejecting qualified applicants
> (false negatives) has a more severe impact on individuals than approving some risky loans
> (false positives). This provides a recall of 0.78 and precision of 0.72, meaning we approve
> more qualified candidates while managing risk. However, I observed that the model shows bias
> against [group], and recommend [mitigation strategy]."

### Step 5: Verify Your Submission Format

Your submission file must:
- Be a CSV file with exactly 2 columns: `id` and `pred`
- Contain 6,750 rows (one for each test record)
- Have predictions as integers: 0 (rejected) or 1 (approved)
- Match the format of `sample_submission.csv`

Example:
```
id,pred
43299,0
15156,1
28745,1
...
```

### Step 6: Submit Your Solution

Send your submission files to the teacher via **Discord**:

**Required files:**
1. **Predictions file**: Your CSV file with predictions (e.g., `yourgroupname_submission.csv`)

**Submission Rules:**
- You have **3 submission attempts** total
- One submission is allowed to return an error (e.g., formatting issue, missing file)
- Only your best-scoring valid submission will count toward your final grade
- Use your attempts wisely: test your code thoroughly before submitting

## Evaluation

Your submission will be evaluated based on three components:

1. **EDA**
2. **Model Pipeline and experimentation**
3. **Model Performance**
4. **Threshold Analysis and justification**
5. **Bias Analysis**

## Tips for Success

### Model Building
- Start with a simple baseline model (e.g., Logistic Regression)
- Handle missing values and outliers carefully
- Try feature engineering to create new meaningful features
- Experiment with different algorithms
- Use cross-validation to avoid overfitting
- Don't forget to apply the same preprocessing to both training and test data

### Bias Analysis
Some suggestions, but you may do different:
- Use confusion matrices for each demographic group to identify disparities
- Calculate metrics like demographic parity, equal opportunity, and equalized odds
- Consider intersectionality (e.g., how does the model perform for young females vs. older males?)
- Think about the historical biases that might be present in the training data
- Document both the biases you find AND potential mitigation strategies

### Threshold Selection
- Don't just use the default 0.5 threshold without justification
- Plot precision-recall curve to visualize tradeoffs
- Consider the cost of false positives vs. false negatives in the loan context
- Think about who is harmed by each type of error
- Test multiple thresholds and document your reasoning for the final choice
- Remember: the "best" threshold depends on business objectives and ethical considerations

Good luck!
