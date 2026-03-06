from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Align input features with training features if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # --- Dynamic Price Range ---
        margin = 20  # you can adjust this margin
        lower_bound = max(0, prediction - margin)
        upper_bound = prediction + margin

        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'price_range': {
                'lower': round(float(lower_bound), 2),
                'upper': round(float(upper_bound), 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
