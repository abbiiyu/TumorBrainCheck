from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from utils import prediksi_tumor, ambil_deskripsi

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = prediksi_tumor(filepath)
            description = ambil_deskripsi(prediction)
            return render_template('prediksi.html', filename=filename, prediction=prediction, description=description)
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
