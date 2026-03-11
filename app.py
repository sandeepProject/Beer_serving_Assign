from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model and column list from your train_model.py
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    model_score =data['best_score']
    model_name=data['final_model_name']
    model_columns = data['columns']
    continents = data['continents']
@app.route('/')
def index():
    # Pass dropdown lists to the HTML landing page
    return render_template('index.html', continents=continents)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the HTML form
    continent = request.form.get('continent')
    beer = float(request.form.get('beer', 0))
    spirit = float(request.form.get('spirit', 0))
    wine = float(request.form.get('wine', 0))

    # Create a DataFrame for the input
    input_df = pd.DataFrame([[continent, beer, spirit, wine]], 
                            columns=['continent', 'beer_servings', 'spirit_servings', 'wine_servings'])
    
    # One-Hot Encode and Align with training columns
    input_encoded = pd.get_dummies(input_df, columns=['continent'])
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

     # Predict
    prediction = model.predict(input_final)[0]
    
    return render_template('index.html', 
                           prediction_text=f'Predicted Alcohol: {prediction:.2f} Litres',
                            continents=continents)

if __name__ == "__main__":
    app.run(debug=True)