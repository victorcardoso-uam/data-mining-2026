# Decision Tree Application Summary – Session 13

## What is a Decision Tree?

A decision tree is a supervised learning model used for predictive modeling. It splits data into smaller groups using decision rules and represents logic in a tree-like structure. The final prediction is produced at the end of each path.

It works like a sequence of if–then rules.

---

## Structure of a Decision Tree

### Root Node
The root node is the first decision in the tree. It is the initial split of the dataset based on the most important feature.

### Internal Nodes
Internal nodes represent decision rules that split the data further.

### Branches
Branches represent the outcome of a decision (True/False or Yes/No).

### Leaf Nodes
Leaf nodes represent the final prediction (a class in classification or a value in regression).

### Path
A path is the sequence of decisions from the root node to a leaf node.

---

## How Decision Trees Split Data

Decision trees split data by selecting features and thresholds that best separate the target classes. For example:

- Is wind speed greater than 8 m/s?
- Is temperature higher than 90°C?

Each question divides the data into two groups, improving the purity of the target variable at each step.

---

## Classification vs Regression Trees

Decision trees can be used for:

### Classification
The output is a category (Yes/No, 0/1, High/Low output).

### Regression
The output is a numerical value (temperature prediction, power output).

In this session, we focused on classification.

---

## Engineering Application Example

In a wind turbine scenario, a decision tree can classify:

- High output
- Low output

This helps in:
- Scheduling maintenance
- Understanding performance conditions
- Supporting engineering decisions

---

## Conclusion

Decision trees are simple, interpretable, and powerful models for predictive analysis. They represent decision logic clearly and are useful for both classification and regression problems.

## Example Application

Decision trees can be applied in many real-world problems. 
For example, in a customer churn prediction project, a decision tree can be used to determine whether a customer is likely to leave a service.

The model analyzes variables such as:
- Customer age
- Subscription type
- Monthly usage
- Payment history

Based on these features, the decision tree splits the data and creates rules that predict if a customer will churn or stay. 
This helps companies identify at-risk customers and take actions to retain them.
