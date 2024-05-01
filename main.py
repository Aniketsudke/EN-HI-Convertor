from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# flask app
app = Flask(__name__)
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained("models/")


def inputEnglish(input_text):
    tokenized = tokenizer([input_text], return_tensors='np')
    out = model.generate(**tokenized, max_length=128)
    with tokenizer.as_target_tokenizer():
        return tokenizer.decode(out[0], skip_special_tokens=True)


# routes


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page


@app.route('/convert', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form.get('valEnglish')

        if input_text == '':
            message = "Please Fill before submit."
            return render_template('index.html', message=message)
        else:
            predicted = inputEnglish(input_text)
            print(predicted)
            return render_template('index.html', predicted=predicted)

    return render_template('index.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
