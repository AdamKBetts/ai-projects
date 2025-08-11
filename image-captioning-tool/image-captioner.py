from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Process the image and generate a caption
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

# Print result
print("Image URL:", img_url)
print("Generated Caption:", caption)