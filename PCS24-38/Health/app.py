import pickle
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# # ----------------------------------------- Data Generation -----------------------------------------

# Set a random seed for reproducibility
np.random.seed(0)

# # Define the number of samples
n_samples = 15000

# Generate synthetic data for the factors
age = np.random.randint(18, 80, n_samples)
height = np.random.uniform(140, 200, n_samples)
weight = np.random.uniform(40, 150, n_samples)
bmi = weight / ((height / 100) ** 2)
sleep_hours = np.random.uniform(4, 12, n_samples)
activity_hours = np.random.uniform(0, 5, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
# Simulate the presence of medical conditions (0 or 1)
medical_conditions = np.random.randint(0, 2, n_samples)
allergies = np.random.randint(0, 2, n_samples)  # Simulate allergies (0 or 1)
diet = np.random.choice(['Balanced', 'Vegetarian', 'Vegan',
                        'Non Vegetarian', 'Paleo', 'Keto'], n_samples)
exercise_routine = np.random.choice(
    ['Regular', 'Irregular', 'Sedentary', 'Athletic'], n_samples)
smoking_habits = np.random.choice(
    ['Non-Smoker', 'Smoker', 'Occasional', 'Former Smoker'], n_samples)
alcohol_consumption = np.random.choice(
    ['Low', 'Moderate', 'High', 'Non-Drinker'], n_samples)
drug_use = np.random.choice(['No', 'Yes', 'Recreational'], n_samples)
stress_management = np.random.choice(
    ['Poor', 'Good', 'Moderate', 'Excellent'], n_samples)
mental_health_history = np.random.choice(
    ['Stable', 'History', 'Depression', 'Anxiety'], n_samples)
family_history = np.random.choice(
    ['No', 'Yes', 'Genetic Conditions'], n_samples)
sleep_schedule = np.random.choice(
    ['Regular', 'Irregular', 'Night Owl', 'Early Bird'], n_samples)
screen_time = np.random.uniform(1, 10, n_samples)
occupation = np.random.choice(['Office Job', 'Manual Labor', 'Student',
                              'Other', 'Healthcare Professional', 'Freelancer'], n_samples)

# Create a Pandas DataFrame
data = pd.DataFrame({
    'Age': age,
    'Height': height,
    'Weight': weight,
    'BMI': bmi,
    'Avg. Hours of Sleep': sleep_hours,
    'Physical Activity Hours': activity_hours,
    'Gender': gender,
    'Medical Conditions': medical_conditions,
    'Allergies': allergies,
    'Diet': diet,
    'Exercise Routine': exercise_routine,
    'Smoking Habits': smoking_habits,
    'Alcohol Consumption': alcohol_consumption,
    'Recreational Drug Use': drug_use,
    'Stress Management': stress_management,
    'Mental Health History': mental_health_history,
    'Family History': family_history,
    'Sleep Schedule': sleep_schedule,
    'Screen Time': screen_time,
    'Occupation': occupation,
})

# Set weights for each factor with respect to the custom health score
weights = {
    'Age': -0.1,
    'BMI': -0.1,
    'Avg. Hours of Sleep': 0.2,
    'Physical Activity Hours': 0.2,
    'Medical Conditions': -0.2,
    'Allergies': -0.05,
    'Screen Time': -0.1,
    'Diet_Keto': 0,
    'Diet_Non Vegetarian': 0,
    'Diet_Paleo': 0,
    'Diet_Vegan': 0.05,
    'Diet_Vegetarian': 0.05,
    'Exercise Routine_Irregular': -0.05,
    'Exercise Routine_Regular': 0.2,
    'Exercise Routine_Sedentary': -0.2,
    'Smoking Habits_Smoker': -0.2,
    'Smoking Habits_Non-Smoker': 0.1,
    'Alcohol Consumption_Low': 0.05,
    'Alcohol Consumption_Moderate': 0,
    'Alcohol Consumption_Non-Drinker': 0.05,
    'Recreational Drug Use_Recreational': -0.1,
    'Recreational Drug Use_Yes': -0.2,
    'Stress Management_Good': 0.2,
    'Stress Management_Moderate': 0,
    'Stress Management_Poor': -0.2,
    'Mental Health History_Depression': -0.2,
    'Mental Health History_History': -0.1,
    'Mental Health History_Stable': 0.1,
    'Family History_No': 0.05,
    'Family History_Yes': -0.05,
    'Sleep Schedule_Irregular': -0.1,
    'Sleep Schedule_Night Owl': -0.05,
    'Sleep Schedule_Regular': 0.1,
    'Occupation_Healthcare Professional': -0.1,
    'Occupation_Manual Labor': -0.05,
    'Occupation_Office Job': 0,
    'Occupation_Other': 0,
    'Occupation_Student': -0.05,
}

# Normalize numeric columns (e.g., Age, Height, Weight) using Min-Max scaling
scaler = MinMaxScaler()
data[['Age', 'Height', 'Weight', 'BMI', 'Avg. Hours of Sleep', 'Physical Activity Hours', 'Screen Time']] = \
    scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI',
                         'Avg. Hours of Sleep', 'Physical Activity Hours', 'Screen Time']])

# Normalize the categorical columns using one-hot encoding
categorical_columns = ['Gender', 'Diet', 'Exercise Routine', 'Smoking Habits', 'Alcohol Consumption',
                       'Recreational Drug Use', 'Stress Management', 'Mental Health History',
                       'Family History', 'Sleep Schedule', 'Occupation']

data = pd.get_dummies(data, columns=categorical_columns, dtype=int)

# Calculate the custom health score based on the weighted factor
sample = [i for i in weights.keys() if i in data.columns]
custom_health_score = sum([data[feature] * weights[feature]
                          for feature in sample])

# Normalize the custom health score to a 0-100 range
custom_health_score = (custom_health_score - custom_health_score.min()) / \
    (custom_health_score.max() - custom_health_score.min()) * 100

# Add the custom health score to the dataset
data['Custom Health Score'] = custom_health_score

# Save the dataset to a CSV file
data.to_csv('health.csv', index=False)

# ----------------------------------------- Data Preprocessing -----------------------------------------

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import plotly.express as px

# Importing the dataset
dataset = pd.read_csv('health.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# ----------------------------------------- Data Splitting ---------------------------------------------

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=0)

# # ----------------------------------------- Model Training ---------------------------------------------

# # Training the Random Forest Regression model on the Training set

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, Y_train)

# # ----------------------------------------- Model Testing ----------------------------------------------

# # Predicting the Test set results
Y_pred = regressor.predict(X_test)

# # ----------------------------------------- Model Evaluation -------------------------------------------

# # Evaluating the Model Performance

print('R2 Score: ', r2_score(Y_test, Y_pred))

# ----------------------------------------- Model Deployment -------------------------------------------


# fig=px.bar(X,x=X[0][::],y=X[0][:],title='The Age and The Number of peapol in The same Age')
# fig.show()
# Saving the model
print(X_train.shape[1])
pickle.dump(regressor, open('health.pkl', 'wb'))
