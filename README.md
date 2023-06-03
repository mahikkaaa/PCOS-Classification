# PCOS Classification Model

This project aims to develop a machine learning model to classify the presence of Polycystic Ovary Syndrome (PCOS) based on various health parameters. The model utilizes a dataset containing information about patients diagnosed with PCOS, including their age, BMI, hormone levels, glucose levels, and other relevant features.

## Dataset

The dataset used for this project can be found at [PCOS Dataset](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos). It contains records of patients diagnosed with PCOS and their corresponding attributes.

Link to my Kaggle Notebook : https://www.kaggle.com/code/mahikkaaa/pcos-prediction-and-analysis

Link to the dataset : https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

## The dataset consists of the following columns:
- Patient File No. : This is the report number which has data for a particular patient
- PCOS : Polycystic ovary syndrome (PCOS) is a hormonal disorder common among women of reproductive age, we would like to determine whether the patient has this syndrome or not
- Age (yrs) : Age of patient in years
- Weight (Kg) : Weight of patient in kg
- Height(Cm) : Height of patient in centimeter
- BMI : Body mass index of the patient
- Blood Group : Blood Group of the patient A+ = 11, A- = 12, B+ = 13, B- = 14, O+ =15, O- = 16, AB+ =17, AB- = 18 (total 8 blood groups)
- Pulse rate(bpm) : It is the heart rate of patient in beats per minute. Resting heart rate for adults ranges from 60 to 100 beats per minute
- RR (breaths/min) : It is the respiration rate. Normal respiration rates for an adult person at rest range from 12 to 16 breaths per minute.
- Hb(g/dl) : Hemoglobin levels in gram per deciliter. For women, a normal level ranges between 12.3 gm/dL and 15.3 gm/dL.
- Cycle(R/I) : ....
- Cycle length(days) : This represents length of menstrual cycle. The length of the menstrual cycle varies from woman to woman, but the average is to have periods every 28 days.
- Marraige Status (Yrs) : Years of marriage
- Pregnant(Y/N) : If the patient is pregnant
- No. of aborptions : No. of aborptions, if any. There are total 541 values out of which 437 patients never had any abortions.
- I beta-HCG(mIU/mL) : this is case 1 of beta hcg
- II beta-HCG(mIU/mL) : this is case 2 of beta hcg (please note: An beta hCG level of less than 5 mIU/mL is considered negative for pregnancy, and anything above 25 mIU/mL is considered positive for pregnancy) (also the unit mIU/mL is mili International Units per miliLiter)
- FSH(mIU/mL) : Its full form is Follicle-stimulating hormone. During puberty: it ranges from 0.3 to 10.0 mIU/mL (0.3 to 10.0 IU/L) Women who are still menstruating: 4.7 to 21.5 mIU/mL (4.5 to 21.5 IU/L) After menopause: 25.8 to 134.8 mIU/mL (25.8 to 134.8 IU/L)
- LH(mIU/mL) : It is Luteinizing Hormone.
- FSH/LH : Ratio of FSH and LH
- Hip(inch) : Hip size in inches
- Waist(inch) : Waist Size in inches
- Waist:Hip Ratio : Waist by hip ratio
- TSH (mIU/L) : It is thyroid stimulating hormone. Normal values are from 0.4 to 4.0 mIU/L
- AMH(ng/mL) : It is Anti-Mullerian Hormone.
- PRL(ng/mL) : This represents Prolactin levels.
- Vit D3 (ng/mL): Vitamin D levels. Normal vitamin D levels in the blood are 20 ng/ml or above for adults.
- PRG(ng/mL): Progesterone levels
- RBS(mg/dl): This value is obtained by doing Random Blood Sugar (RBS) Test.
- Weight gain(Y/N): Is there been a weight gain
- hair growth(Y/N): Is there been a hair growth
- Skin darkening (Y/N): Skin darkening issues
- Hair loss(Y/N): hair loss issues
- Pimples(Y/N): pimples issues
- Fast food (Y/N): is fast food part of you diet
- Reg.Exercise(Y/N): do you do exercises on a regular basis
- BP _Systolic (mmHg): Systolic blood pressure, measures the pressure in your arteries when your heart beats.
- BP _Diastolic (mmHg): Diastolic blood pressure, measures the pressure in your arteries when your heart rests between beats.
- Follicle No. (L): Follicles number in the left side
- Follicle No. (R): Follicles number in the right side
- Avg. F size (L) (mm): Average Follicle size in the left side in mm
- Avg. F size (R) (mm): Average Follicle size in the right side in mm
- Endometrium (mm): Size of Endometrium in mm

## Model Training

The PCOS classification model is trained using the scikit-learn library in Python. The following steps are followed for model training:

1. Load the dataset: The dataset is loaded into the model, and the input features (X) and the target variable (y) are extracted.

2. Data preprocessing: The dataset may require preprocessing steps such as handling missing values, scaling numerical features, or encoding categorical variables.

3. Splitting the data: The dataset is split into training and testing sets using the train_test_split function. This allows us to evaluate the model's performance on unseen data.

4. Model selection: Various classification algorithms can be considered for building the PCOS classification model. Common choices include logistic regression, decision trees, random forests, or support vector machines. The choice of the algorithm depends on the specific requirements and characteristics of the dataset.

5. Model training: The selected classification algorithm is trained on the training set using the fit method.

6. Model evaluation: The trained model is evaluated using appropriate evaluation metrics such as accuracy, precision, recall, or F1 score. The model's performance on the testing set is assessed to determine its effectiveness in classifying PCOS.

7. Hyperparameter tuning: Fine-tuning of the model's hyperparameters may be performed to optimize its performance. Techniques such as cross-validation or grid search can be employed to find the best combination of hyperparameters.

## Machine Learning Models
We will first import all the libraries required for creating the models, such as SkLearn, Numpy, and Pandas.
In this dataset we have a mixture of category and numerical data in the dataset. We need to transform the categorical data into numerical data in order to create prediction models.

After conversion we'll build the classification models using the following:
### Linear Models
  1. Logistic Regression (accuracy - 0.84)
  2. SVM (accuracy - 0.67)

### Non Linear Models
  1. Guassian Naive Bayes (accuracy - 0.83)
  2. Random Forest Classifier (accuracy - 0.88)
  3. K Neighbors Neighbor Classifier (accuracy - 0.67)
  4. Decision Tree Classifier (accuracy - 0.80)
  5. XGBoost Classifier (accuracy - 0.86)

## Dependencies

The following Python libraries are required to run the PCOS classification model:

- sklearn
- pandas
- numpy


## Usage

To use the PCOS classification model, follow these steps:

1. Clone the repository or download the source code.

2. Ensure that the dataset is available and stored in the appropriate directory.

3. Open the main Python script or Jupyter Notebook containing the model implementation.

4. Modify the file paths or dataset loading code to point to the correct location of the dataset.

5. Run the script or notebook to train and evaluate the model.

## Results

The performance of the PCOS classification model is evaluated using standard evaluation metrics such as accuracy, precision, recall, and F1 score. The results are presented in a

 clear and understandable format, providing insights into the model's effectiveness in classifying PCOS.

## Conclusion

The PCOS classification model developed in this project demonstrates the potential of machine learning techniques in diagnosing PCOS based on patient health parameters. The model's accuracy and performance metrics indicate its ability to classify PCOS cases accurately. However, it's important to note that the model's predictions should not replace medical diagnosis and should be used as a complementary tool for healthcare professionals.

## Future Work

- Further exploration of different machine learning algorithms and techniques to improve the model's performance.
- Integration of additional features or data sources to enhance the model's accuracy.
- Deployment of the model as a web application or API for easy access and usage.
- Collaboration with medical professionals to validate the model's predictions and refine its performance.


Feel free to contribute, modify, or use this project as a starting point for your own PCOS classification model.
