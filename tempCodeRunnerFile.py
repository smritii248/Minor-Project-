from flask import Flask, request, jsonify, render_template
import requests
import markdown2
from PIL import Image
import pytesseract
import io

app = Flask(__name__)

# FastAPI backend URL (keep same)
FASTAPI_URL = 'http://127.0.0.1:8000'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.get_json()
    medical_text = data.get('medical_text')
    if not medical_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Forward to your FastAPI endpoint (adjust endpoint name if needed)
        response = requests.post(f"{FASTAPI_URL}/simplify_text", json={'input': medical_text})
        response.raise_for_status()
        result = response.json()

        # Optional: render markdown if your FastAPI returns raw text
        if 'simplified_explanation' in result:
            result['simplified_explanation'] = markdown2.markdown(result['simplified_explanation'])

        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Backend error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional: Keep OCR upload if you want (can add separate route/button later)
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    try:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        # Then call simplify internally or return text for frontend to send
        return jsonify({'extracted_text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)