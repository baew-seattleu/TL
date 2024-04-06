# <p align="center"> Transfer Learning for Asthma Prediction </p>

<div align="center">
 <a href="https://github.com/jay-singhvi/Research-Project">
    <img src="img/logo.png" alt="logo" width="80" height="80">
  </a>

  <h3 align="center">Transfer Learning using Ensemble Models to Predict Health Risk in Asthma Patients</h3>
  
  <p align="center">
    Seattle University: Data Science Research
    <br />
    <br />
    <a href="https://github.com/baew-seattleu">Wan Bae</a>
    <br /> 
    <a href="https://github.com/madhuroopa">Madhuroopa Irukulla</a><br>
    <a href="https://github.com/jay-singhvi">Jay Singhvi</a>
  </p>
</div>

This repository contains Python code for predicting asthma risk in patients using transfer learning and ensemble modeling techniques. The code is designed to work with asthma patient data.

### Approach

The approach consists of two main steps:

1. **Population Modeling**: Generalized models are trained on data from all patients to capture common patterns and characteristics related to asthma risk.

2. **Individual Modeling**: The population models are fine-tuned and transferred to individual patient data, allowing for personalized predictions and improved accuracy.

Additionally, ensemble models are created by combining the individual neural network models with conventional machine learning algorithms like Decision Trees, K-Nearest Neighbors, Support Vector Machines, Logistic Regression, and Naive Bayes. This ensemble approach leverages the strengths of multiple models to enhance prediction performance.

### Code Structure

The code is organized into the following main components:

1. **Data Loading and Preprocessing**: The code includes functions to load and preprocess the asthma patient data, handling tasks such as data cleaning, feature selection, and data augmentation using techniques like ADKNN, GammaSmoteBoost, and SMOTEBoostCC.

2. **Population Modeling**: This module trains generalized neural network models on the entire patient dataset, using techniques like k-fold cross-validation and hyperparameter tuning to optimize performance.

3. **Individual Modeling**: The population models are fine-tuned and transferred to individual patient data, allowing for personalized predictions. This module includes functions for training and evaluating individual models.

4. **Ensemble Modeling**: This module combines the individual neural network models with conventional machine learning algorithms to create ensemble models, leveraging the strengths of multiple modeling approaches.

5. **Evaluation and Visualization**: The code includes functions for evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). Additionally, visualizations like training curves and confusion matrices are provided to aid in model analysis and interpretation.

### Key Findings

Through the application of transfer learning and ensemble modeling techniques, the code achieves improved prediction accuracy for asthma risk compared to traditional modeling approaches. The ensemble models leverage the strengths of both neural networks and conventional machine learning algorithms, resulting in robust and reliable predictions.

Furthermore, the individual modeling step allows for personalized predictions tailored to each patient's unique characteristics, enabling more targeted and effective treatment plans.

### Getting Started

To get started with the code, follow these steps:

1. Clone the repository: `git clone https://github.com/jay-singhvi/Research-Project.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Prepare your asthma patient data in the appropriate format.
4. Run the notebooks, adjusting the parameters and file paths as needed.

Transfer Learning has 3 parts - 
1. Training - Population Models - Here we create generalized models for all patients     <br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%201%20-%20TL_Model/TL_Population_Model.ipynb)     <br />     <br />
2. Re-Training - Individual Models - Here we use existing models to learn about a specific patient at a time     <br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%201%20-%20TL_Model/TL_Individual_Model.ipynb)     <br />
   2a. Epoch Analysis for Models to find which combination is best and consistently amongst highest values in Specificity     <br />
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%202%20-%20TL_Model_Eval_Analysis/TL_Epochs_Analysis.ipynb)     <br />     <br />
3. Ensemble Modelling - By combining them to base models (conventional models like Decision Tree, KNearest Neighbor, Support Vector Machine, Logistic Regression and Naive Bayes) to predict Asthma in Patients.    <br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%204%20-%20TL%2BBase_Model/TL_BaseModel.ipynb) <br />
    3a. Epoch Analysis for Ensemble Models to find which combination is best and consistently amongst highest values in Specificity     <br />
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%205%20-%20TL_Model_Eval_Analysis/TL_Epochs_Analysis.ipynb)     <br />     <br />
### Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).

### Acknowledgments

We would like to thank Seattle University and the Data Science program for their support and guidance throughout this research project.
