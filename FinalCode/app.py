from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Define the path where the models are saved
model_save_path = 'Models'  

def predict_movie_success(name, budget, gross, gross_income_dollar, director, pop_rank_director, producer, pop_rank_producer,
                          composer, pop_rank_composer, actor_1_name, pop_rank_actor_1, actor_2_name, pop_rank_actor_2,
                          actor_3_name, pop_rank_actor_3, total_rank, total_fb_like):
    """
    Predicts whether a movie is a hit or flop using three pre-trained models.

    Args:
        Various movie-related input values.

    Returns:
        dict: Predictions from the three models (SVM, Logistic Regression, and Naive Bayes).
    """


    # Create a DataFrame from the input data
    input_data = {
        'Name': name,
        'budget': budget,
        'Gross': gross,
        'gross_income_doller': gross_income_dollar,
        'directors': director,
        'pop_rank_director': pop_rank_director,
        'producer': producer,
        'pop_rank_producer': pop_rank_producer,
        'composer': composer,
        'pop_rank_composer': pop_rank_composer,
        'actor_1_name': actor_1_name,
        'pop_rank_actor_1': pop_rank_actor_1,
        'actor_2_name': actor_2_name,
        'pop_rank_actor_2': pop_rank_actor_2,
        'actor_3_name': actor_3_name,
        'pop_rank_actor_3': pop_rank_actor_3,
        'total_rank': total_rank,
        'total_fb_like': total_fb_like
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Load the trained models
    svm_model = joblib.load(f"{model_save_path}/SVM.joblib")
    logistic_regression_model = joblib.load(f"{model_save_path}/Logistic_Regression.joblib")
    naive_bayes_model = joblib.load(f"{model_save_path}/Naive_Bayes.joblib")

    # Make predictions using the models
    svm_prediction = svm_model.predict(input_df)[0]
    logistic_regression_prediction = logistic_regression_model.predict(input_df)[0]
    naive_bayes_prediction = naive_bayes_model.predict(input_df)[0]

    # Return predictions as a dictionary
    predictions = {
        'SVM': 'Hit' if svm_prediction == 1 else 'Flop',
        'Logistic Regression': 'Hit' if logistic_regression_prediction == 1 else 'Flop',
        'Naive Bayes': 'Hit' if naive_bayes_prediction == 1 else 'Flop'
    }

    return predictions

# Define the route to display the input form
@app.route('/')
def input_form():
    return render_template('index.html')

# Define the route to handle form submission and display predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    name = request.form['name']
    budget = float(request.form['budget'])
    gross = float(request.form['gross'])
    gross_income_dollar = float(request.form['gross_income_dollar'])
    director = request.form['director']
    pop_rank_director = int(request.form['pop_rank_director'])
    producer = request.form['producer']
    pop_rank_producer = int(request.form['pop_rank_producer'])
    composer = request.form['composer']
    pop_rank_composer = int(request.form['pop_rank_composer'])
    actor_1_name = request.form['actor_1_name']
    pop_rank_actor_1 = int(request.form['pop_rank_actor_1'])
    actor_2_name = request.form['actor_2_name']
    pop_rank_actor_2 = int(request.form['pop_rank_actor_2'])
    actor_3_name = request.form['actor_3_name']
    pop_rank_actor_3 = int(request.form['pop_rank_actor_3'])
    total_rank = int(request.form['total_rank'])
    total_fb_like = int(request.form['total_fb_like'])

    # Call the predict_movie_success function to get predictions
    predictions = predict_movie_success(
        name, budget, gross, gross_income_dollar, director,
        pop_rank_director, producer, pop_rank_producer,
        composer, pop_rank_composer, actor_1_name, pop_rank_actor_1,
        actor_2_name, pop_rank_actor_2, actor_3_name, pop_rank_actor_3,
        total_rank, total_fb_like
    )

    
    # Render the results page with the predictions
    return render_template('result.html', predictions=predictions)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
