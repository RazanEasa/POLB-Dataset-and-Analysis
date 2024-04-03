# POLB-Dataset-and-Analysis

This tool allows for the evaluation of various classifiers using a comprehensive set of metrics such as accuracy, precision, recall, specificity, and F1 score.

## Setup

1. **Clone Repository**: Clone this repository to your local machine using the following command:

    ```bash
    git clone https://github.com/RazanEasa/POLB-Dataset-and-Analysis.git
    ```

2. **Install Dependencies**: Ensure you have Python installed on your machine. Install the required Python libraries by running:

    ```bash
    pip install pandas numpy matplotlib seaborn tabulate scikit-learn xgboost
    ```

3. **Dataset**: Place your dataset in CSV format in the same directory as the code file. Ensure that the dataset includes a column labeled 'class' for classification.

## Usage

1. **Change Directory**, change your current working directory to the `POLB-Dataset-and-Analysis` directory by executing the following command in your terminal:

    ```bash
    cd POLB-Dataset-and-Analysis
    ```

2. **Run Code**: Execute the Python script `Classificationcode.py` to run the classifier evaluation tool. This script evaluates various classifiers on the provided dataset and outputs the results to the console.

    ```bash
    python Classificationcode.py
    ```

3. **Interpret Results**: The tool will display a table summarizing the performance of each classifier in terms of accuracy, precision, recall, specificity, and F1 score.

4. **View Results**: Review the classification performance of each classifier and compare their effectiveness in handling your dataset.

## Customization

- **Dataset**: If using a different dataset, ensure it is in CSV format and contains a column labeled 'class' for classification.
  
- **Classifier Parameters**: Adjust the parameters of the classifiers within the script to optimize performance for your specific dataset.



Feel free to reach out if you have any questions or encounter any issues!


