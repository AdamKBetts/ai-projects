import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model and processor once when the app starts
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# The root URL of our application
@app.route('/')
def index():
    return render_template('index.html')

# The API endpoint for generating a caption
@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        # Save the file securely and open it with Pillow
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        raw_image = Image.open(filepath).convert('RGB')

        # Generate the caption
        text = "a photograph of "
        inputs = processor(raw_image, text, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Clean up the uploaded file
        os.remove(filepath)

        return jsonify({'caption': caption})
    
if __name__ == '__main__':
    app.run(debug=True)