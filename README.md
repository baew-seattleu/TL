<br />
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

Transfer Learning has 2 parts - 

1. Learning - Population Models - Here we create generalized models for all patients     <br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%201%20-%20TL_Model/TL_Population_Model.ipynb)
2. Transfer Learning - Individual Models - Here we use existing models to learn about a specific patient at a time     <br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%201%20-%20TL_Model/TL_Individual_Model.ipynb)
3. We have used these Neural Network - Individual Models to create Ensemble Models by combining them to base models (conventional models like Decision Tree, KNearest Neighbor, Support Vector Machine, Logistic Regression and Naive Bayes) to predict Asthma in Patients.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baew-seattleu/TL/blob/main/Models/classification/Step%204%20-%20TL%2BBase_Model/TL_BaseModel.ipynb)
