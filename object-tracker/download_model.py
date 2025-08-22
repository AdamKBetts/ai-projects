import tensorflow as tf
import tarfile
import os
# The URL for the pre-trained SSD MobileNet V2 model
download_base = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
tar_file_name = model_name + '.tar.gz'
model_dir = model_name + '/saved_model'

# Download the model
print(f'Downloading model: {tar_file_name}')
download_path = tf.keras.utils.get_file(
    fname=tar_file_name,
    origin=download_base + tar_file_name,
    cache_dir=os.path.abspath('.'),
    cache_subdir='model'
)

# Extract the tar.gz file
print(f'Extracting model to: {model_name}')
with tarfile.open(download_path, 'r:gz') as tar:
    tar.extractall()

print("Model download and extraction complete!")