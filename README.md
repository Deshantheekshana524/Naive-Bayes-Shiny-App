# Naive Bayes Classifier Shiny App

This is an interactive web application built with R and Shiny that allows users to upload custom datasets, visualize the relationship between target classes and features, train a Naive Bayes classification model, and evaluate performance.

## ✨ Features

- **Upload & Interactive Data Prep:**
  - Upload custom CSV datasets.
  - Dynamically select target variables and features for exploratory visualizations.
  - Automatically or manually configure categorical vs. continuous variable types.
  - Custom training/testing split ratio slider with reproducibility (fixed seed) toggle.

- **Visual Explorations:**
  - **Dataset Preview:** Interactive data table to inspect raw data.
  - **Target vs. Feature Plots:** Automatically switches between grouped bar charts (for categorical features) and boxplots (for continuous features).

- **Comprehensive Performance Metrics:**
  - **Value Boxes:** Real-time metrics tracking overall Accuracy, Precision, Recall, and F1 Score.
  - **Confusion Matrix:** Beautiful heatmap visualization of model predictions against actual test labels.
  - **Per-Class Metrics Table:** Detailed breakdown including Specificity, Balanced Accuracy, Negative Predictive Value (NPV), Prevalence, and Support.

- **Inference & Predictions Panel:**
  - **Batch Scoring via CSV:** Upload new datasets (without target columns), generate batch predictions with class probabilities, and download results as a CSV.
  - **Manual Entry Mode:** Automatically generated form inputs based on your training data schema to test single observations on the fly with live probability breakdowns.

---

## Access the Application
You can access and use the live application by clicking the link below:

[Click Here](https://your-shiny-app-url-here.shinyapps.io/naive-bayes-app/)
