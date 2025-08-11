Project 3: Image Captioning and Analysis Tool

This project demonstrates a Multimodal AI application that can generate a descriptive caption for any given image. It showcases the use of Vision-Language Models (VLMs), which are a powerful class of AI that understands both visual and textual data. This tool can be used for a wide range of tasks, from automated image tagging to accessibility enhancements.

Features ✨
Image Analysis: The tool processes an image to understand its content.

Automatic Captioning: It generates a coherent and detailed natural language caption based on the image.

Hugging Face Integration: Uses a pre-trained BLIP model from the Hugging Face ecosystem, demonstrating proficiency with state-of-the-art open-source AI models.

Flexible Input: Supports image input from both URLs and local files.

Core Components 🧩
Hugging Face Transformers: Provides the core functionality for the Vision-Language Model (BLIP).

Pillow: A Python library for handling and manipulating image data.

requests: Used to fetch images from a specified URL.

Installation 🛠️
Clone the repository and navigate to the project directory:

Bash

git clone https://github.com/AdamKBetts/ai-projects.git
cd ai-projects/image-captioning-tool
Install the required Python libraries:

Bash

pip install transformers Pillow requests
Usage ▶️
To run the image captioning tool, ensure you are in the image-captioning-tool directory with your virtual environment active, and then execute the script.

Bash

python image_captioner.py
The first time you run this, the script will download the pre-trained BLIP model (approximately 400MB). After the download is complete, it will generate and print a caption for the default image.

Customizing the Image
You can easily change the image the script analyzes by editing the image_captioner.py file.

To use a local image file:

Place your image file (e.g., my_image.jpg) in the same directory as the script.

Change this line:

Python

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
to this:

Python

image_path = 'my_image.jpg'
raw_image = Image.open(image_path).convert('RGB')
To use a different image URL:

Simply replace the URL string in the img_url variable with a new one.

Python

img_url = 'https://link-to-your-new-image.jpg'
