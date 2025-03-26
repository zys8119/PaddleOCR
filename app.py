import os
os.system('pip install paddlepaddle')
os.system('pip install paddleocr')
import requests
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr

def download_image(url, save_path):
    """
    Download an image from a specified URL and save it to the specified path
    
    Args:
        url (str): URL of the image
        save_path (str): Path to save the image
        
    Returns:
        bool: True if download is successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Image successfully downloaded and saved as: {save_path}")
            return True
        else:
            print(f"Download failed, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        return False

# Download example image from GitHub
image_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v2.8.0/doc/imgs_en/img_12.jpg"
download_image(image_url, "example.jpg")

def inference(img, lang):
	ocr = PaddleOCR(use_angle_cls=True, lang=lang,use_gpu=False)
	img_path = img  
	result = ocr.ocr(img_path, cls=True)[0]
	image = Image.open(img_path).convert('RGB')
	boxes = [line[0] for line in result]
	txts = [line[1][0] for line in result]
	scores = [line[1][1] for line in result]
	im_show = draw_ocr(image, boxes, txts, scores,
			   font_path='./simfang.ttf')
	im_show = Image.fromarray(im_show)
	im_show.save('result.jpg')
	return 'result.jpg'

title = 'PaddleOCR'
description = 'Gradio demo for PaddleOCR. PaddleOCR demo supports Chinese, English, French, German, Korean and Japanese. To use it, simply upload your image and choose a language from the dropdown menu, or click one of the examples to load them. Read more at the links below.'
article = "<p style='text-align: center'>Awesome multilingual OCR toolkits based on PaddlePaddle <a href='https://github.com/PaddlePaddle/PaddleOCR'>Github Repo</a></p>"
examples = [['example.jpg','en']]
css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(
    inference,
    [gr.Image(type='filepath', label='Input'), gr.Dropdown(choices=['ch', 'en', 'fr', 'german', 'korean', 'japan'], value='en', label='language')],
    gr.Image(type='filepath', label='Output'),
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=css
    ).launch(debug=True)
