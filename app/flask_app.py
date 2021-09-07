from flask import Flask, request
import pickle






app = Flask(__name__)


pickle_in = open('rf_classifier.pkl', 'rb')
rf_classifier = pickle.load(pickle_in)


@app.route('/')
def first_func():
    return 'Hello all!'


@app.route('/predict', methods=['Get'])
def predict_bank_note():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = rf_classifier.predict([[variance,skewness,curtosis, entropy]])
    return ' result is ' + str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)