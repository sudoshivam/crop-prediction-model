  

# Crop Recommendation Model
A machine learning model to analyze soil parameters and recommend suitable crops. It uses Logistic Regression to predict crops based on the following 7 input parameters.

 1. Nitrogen (N) 
 2. Potassium (P) 
 3. Phosphorus (K) 
 4. Temperature 
 5. Humidity 
 6. pH Value 
 7. Rainfall
 
- The N-P-K values are ratio between Nitrogen, Phosphorous and Potassium. This means if soil contains 3% nitrogen (N), 2% phosphorus (P) and 5% potassium (K) then its NPK value is 3-2-5. 
- Temperature, Humidity and Rainfall data can be acquired from weather portal or IoT sensors.
- pH value is used to measure if soil is naturally acidic or alkaline. This data is acquired by testing soil in lab.

### Dataset Sample
![Dataset Sample](https://raw.githubusercontent.com/sudoshivam/crop-prediction-model/main/static/images/01.png)
## Usage
 **Enter all parameters in respective input boxes (Make sure all input values are rounded to nearest integer).**
 ![Enter Parameters](https://raw.githubusercontent.com/sudoshivam/crop-prediction-model/main/static/images/02.png)

**Click on Predict.** 
![enter image description here](https://raw.githubusercontent.com/sudoshivam/crop-prediction-model/main/static/images/03.png)
The model will analyze input parameters and predict the most suitable crop for given soil condition.

## Run Locally

 - Make sure git is installed. 
 - Clone the project by running following command in cmd or terminal: `git clone https://github.com/sudoshivam/crop-prediction-model.git` or you can simply download the entire repo as zip.
 - Install Anaconda or miniconda and create an environment with Python 3.6. Refer [this tutorial](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on how to create an environment in conda.
 - Activate environment.
 - Install required libraries by running following commands:
	 - Scikit-Learn `pip install -U scikit-learn`.
	- NumPy `pip install numpy`.
	- Flask `pip install Flask`.
 - Open new anaconda prompt in project folder and run following command:
  `python app.py`
  - Now project will start running in the terminal. Copy the URL provided in terminal and open it into a web browser to use the application. 
 

