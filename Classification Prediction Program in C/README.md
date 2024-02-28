# Classification Prediction Application in C

## Overview
This repository contains a classification prediction application written in C, designed to predict the class of data points. The program utilizes three optimization algorithms: Adam, Stochastic Gradient Descent (SGD), and Gradient Descent (GD) to train and classify data points into predefined classes.

## Problem Definition
The classification prediction application addresses the task of accurately categorizing data points into two predefined classes based on given features. This is a common problem in machine learning, where the goal is to develop a model that can generalize patterns in the data and make accurate predictions on new, unseen data.

##Algorithms Used
The application employs the following optimization algorithms:

```1. Gradient Descent
*Basic optimization algorithm for updating model parameters.
*Iteratively minimizes the model's loss function to improve accuracy.
```
```2. Stochastic Gradient Descent (SGD)
*Optimizes using a single data point per iteration.
*Suitable for large datasets as it processes one data point at a time.
```
```3. Adam Optimizer
*Utilizes adaptive moment estimation for efficient optimization.
*Effective in scenarios with varying or sparse data.
```