# Final Analytics Project

Our project utilizes a Device Addiction data to predict whether a person will be addicted to their device or not.

[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)

## Background info

The data set that we will be examining is on cellphone addiction. The main question we want to answer is: **"Given information about a person and their cell phone usage can we predict if they are addicted to their cell phone?"** The data set has 7500 rows with different features including age, gender, screen time, stress levels, etc. It also has a category which specifies if the person is addicted or not. We use this as the true classes that we compare our predicted classes to. This dataset is a simulation of real-world behavioral data and consists of synthetic user records which are designed for predicting digital addiction labels.

We decided on this data set because it highlights a large device addiction problem in society, specifically for younger adults. By building and evaluating models on this data set, we will be able to determine the specific metrics that contribute most to being addicted to cell phones. By discovering the specific metrics that contribute most to addiction, we are able to help prevent cellphone addiction by warning people on the biggest causes of addiction, thus enabling them to do their best to avoid cellphone addiction themselves.

## Model Evaluation

### Decision Tree Model
The final decision tree model achieved a balanced accuracy of 95.85%, an accuracy of 94.13%, and a recall of 91.71%. Since false negative is the number that we are trying to reduce, the recall is looking good and and the model is performing well comparing to the prevalence of 70.77%. This if further reflected by the AUC score of 0.9894. It is worth noting the the model actually has a precision of 1, so there were no identified cases of false positives. In relation to the problem, two variables dominates in importance, gaming hour and work study hour, which came rather unexpected, but offers valuable knowledge towards answering our question: given the gaming and work study hours, we can reliably determine whether someone is addicted.

## Conclusions


## Team Contribution
