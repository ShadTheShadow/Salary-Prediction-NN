# 💼 Salary Classification with Dense Neural Networks

## Overview (check out the [full workbook](https://github.com/ShadTheShadow/Salary-Prediction-NN/blob/main/ML4ER_FinalProject_EvanWilliams%20(4-17-25).ipynb))

This project explores the use of a Dense Neural Network (Dense-NN) machine learning model to predict whether an individual's salary exceeds $50k/year. The classification problem is approached using Python, TensorFlow, and MAST-ML for feature engineering and model evaluation.

The target variable is binary (above or below $50k/year), and the model is trained using binary crossentropy loss to optimize performance. The goal is to experiment with different model complexities and features to achieve the most accurate results.

## Tools & Technologies

- **Python**
- **TensorFlow / Keras**
- **MAST-ML** for model comparison and preprocessing
- **Pandas**, **NumPy**, **Matplotlib** for data analysis and visualization

## Dataset

[Original dataset](https://www.kaggle.com/datasets/uciml/adult-census-income/data)

The original data is cleaned and preprocessed in the workbook to ensure the best results for the neural network. The data includes features such as...

- Age
- Education
- Occupation
- Hours per week
- Relationship status
- Native country
- Income
- etc.

Simplified model feature distributions:

![Age distro](https://github.com/user-attachments/assets/b655de28-2f1e-4374-8e7d-4f90f64a8abd)
![Income distro](https://github.com/user-attachments/assets/bf6fa8e2-0adc-4b1e-89f9-dd5edfaf8b78)


## Approach

1. **Preprocessing**: Categorical encoding, missing value handling, and feature scaling.
2. **Modeling**: Built two Dense Nerual Network archictectures, one with two features, and the other with the full 12
3. **Evaluation**: Compared models using F1 score, accuracy, and loss plots across training and validation sets.

## Simplified model results (age -> salary)

![Simplified model](https://github.com/user-attachments/assets/786bc21d-092a-4ae3-9d75-9d24a6f1a805)

- Ending with an F1 score of 0.59 our model is not the most accurate, however the training of the model went very well
- This model only uses age to predict salary, so given the circumstances, it has good performance

## Full model results (all features -> salary)

Added all 12 features with hopes of increasing F1 score

![Full data - cleaned](https://github.com/user-attachments/assets/776c1aba-0b93-42d9-b339-92d4e547e673)


![Full model](https://github.com/user-attachments/assets/6cc66dd3-fcfd-4fe3-8292-c6a56052fdd8)

- Ending with an F1 score of 0.76 our model has significantly improved
- The training for this particular model was a little troublesome, likely because of the wide variety of features available sometimes providing seperate information.
- With further tuning the model could vastly improve, however this accomplishes our initial goal of a very high F1 score


## Conclusion

Through systematic tuning, the final Dense-NN model achieved a strong increase in both F1 score and accuracy. The transition from a simple to a more complex architecture proved to be valuable, despite added training variability. This highlights the importance of model complexity and thoughtful feature engineering in binary classification tasks.


## Want to run the code?

To run the code yourself:
- Download the .ipynb file from git or clone to a local repo
- Upload the file to google colab for best results
