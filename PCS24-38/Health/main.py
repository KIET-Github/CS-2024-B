import logging
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


class HealthScore:
    def __init__(self) -> None:
        pass

    @staticmethod
    def decode(sorted_data):
        array = []
        age = int(sorted_data[0])
        age_new = (age - 18) / (80 - 18)
        array.append(age_new)  # Age is normalized

        height = int(sorted_data[1])
        height_new = (height - 140) / (200 - 140)
        array.append(height_new)  # Height is normalized

        weight = int(sorted_data[2])
        weight_new = (weight - 40) / (150 - 40)
        array.append(weight_new)  # Weight is normalized

        bmi = weight / ((height / 100) ** 2)
        array.append(bmi)  # BMI is calculated

        sleep_hours = int(sorted_data[4])
        sleep_hours_new = (sleep_hours - 4) / (12 - 4)
        array.append(sleep_hours_new)  # Sleep hours is normalized

        activity_hours = int(sorted_data[5])
        activity_hours_new = (activity_hours - 0) / (5 - 0)
        array.append(activity_hours_new)  # Activity hours is normalized

        if (sorted_data[6] == "Yes"):  # Medical Conditions is one hot encoded
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(1)

        if (sorted_data[7] == "Yes"):  # Allergies is one hot encoded
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(1)

        screen_time = int(sorted_data[8])  # Screen time is normalized
        screen_time_new = (screen_time - 1) / (10 - 1)
        array.append(screen_time_new)

        if (sorted_data[9] == "Male"):  # Gender is one hot encoded
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(1)

        if (sorted_data[10] == "Balanced"):  # Diet is one hot encoded
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[10] == "Keto"):
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[10] == "Non Vegetarian"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[10] == "Paleo"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[10] == "Vegan"):
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[11] == "Athletic"):  # Exercise Routine is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(1)
        elif (sorted_data[11] == "Irregular"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[11] == "Regular"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[12] == "Former Smoker"):  # Smoking Habits is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[12] == "Non-Smoker"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[12] == "Occasional"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[13] == "High"):  # Alcohol Consumption is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[13] == "Low"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[13] == "Moderate"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[14] == "No"):  # Recreational Drug Use is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[14] == "Recreational"):
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[15] == "Excellent"):  # Stress Management is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[15] == "Good"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[15] == "Moderate"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[16] == "Anxiety"):  # Mental Health History is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[16] == "Depression"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[16] == "History"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[17] == "Genetic Conditions"):  # Family History is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[17] == "Yes"):
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[18] == "Early Bird"):  # Sleep Schedule is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[18] == "Night Owl"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[18] == "Regular"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        if (sorted_data[19] == "Freelancer"):  # Occupation is one hot encoded
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[19] == "Healthcare Professional"):
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[19] == "Manual Labor"):
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
            array.append(0)
        elif (sorted_data[19] == "Office Job"):
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
            array.append(0)
        elif (sorted_data[19] == "Other"):
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)
            array.append(0)
        else:
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(0)
            array.append(1)

        return array


# Data Order
format = {
    0: 'Age',
    1: 'Height',
    2: 'Weight',
    3: 'BMI',
    4: 'Avg. Hours of Sleep',
    5: 'Physical Activity Hours',
    6: 'Medical Conditions',
    7: 'Allergies',
    8: 'Screen Time',
    9: 'Gender',
    10: 'Diet',
    11: 'Exercise Routine',
    12: 'Smoking Habits',
    13: 'Alcohol Consumption',
    14: 'Recreational Drug Use',
    15: 'Stress Management',
    16: 'Mental Health History',
    17: 'Family History',
    18: 'Sleep Schedule',
    19: 'Occupation',
}

# userData = [23, 182, 75, 7, 2, "Male", "No", "No", "Balanced", "Regular", "Non-Smoker",
#             "Low", "No", "Good", "Anxiety", "No", "Early Bird", 5, "Healthcare Professional"]

# Home Route
# {Physical Activity Hours: 1, Medical Conditions: Yes, Recreational Drug Use: Yes, Stress Management: Good, Exercise Routine: Regular, Gender: Male, Screen Time: 6, Height: 150, Avg. Hours of Sleep: 8, Age: 20, Mental Health History: Depression, Smoking Habits: Smoker, Alcohol Consumption: High, Weight: 70, Allergies: Yes, Sleep Schedule: Irregular, Occupation: Student, Diet: Vegetarian, Family History: Genetic Conditions}


@app.route('/')
def home():
    return 'Welcome to the Health Score Prediction API'


# Predict Custom Health Score Route
@app.route('/predict_custom_health_score', methods=['POST'])
def predict_custom_health_score():
    # Load the trained model
    loaded_regressor = pickle.load(open('health.pkl', 'rb'))
    try:
        data = request.get_json()
        print(data)
        res = [0] * 20
        for i in range(20):
            if format[i] == "BMI":
                res[i] = (float(data['Weight']) /
                          ((float(data['Height']) / 100) ** 2))
            else:
                res[i] = (data[format[i]])
        obj = HealthScore()
        array = obj.decode(res)
        b = np.array(array).reshape(1, -1)
        custom_health_score = loaded_regressor.predict(
            b)  # Use the loaded model for prediction

        val = custom_health_score[0]
        return jsonify({'healthScore': val})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': 'An internal server error occurred.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
