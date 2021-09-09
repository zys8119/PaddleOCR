import os
os.system('pip install paddlepaddle')
os.system('pip install paddleocr')
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr
import torch

torch.hub.download_url_to_file('https://i.imgur.com/aqMBT0i.jpg', 'example.jpg')

def inference(img, lang):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang,use_gpu=False)
    img_path = img.name
    result = ocr.ocr(img_path, cls=True)
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores,
                       font_path='simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    return 'result.jpg'

title = 'PaddleOCR'
description = 'Gradio demo for PaddleOCR. PaddleOCR demo supports Chinese, English, French, German, Korean and Japanese.To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'
article = "<p style='text-align: center'><a href='https://www.paddlepaddle.org.cn/hub/scene/ocr'>Awesome multilingual OCR toolkits based on PaddlePaddle （practical ultra lightweight OCR system, support 80+ languages recognition, provide data annotation and synthesis tools, support training and deployment among server, mobile, embedded and IoT devices）</a> | <a href='https://github.com/PaddlePaddle/PaddleOCR'>Github Repo</a></p>"
examples = [['example.jpg']]
css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(
    inference,
    [gr.inputs.Image(type='file', label='Input'),gr.inputs.Dropdown(choices=['ch', 'en', 'fr', 'german', 'korean', 'japan'], type="value", default='en', label='language')],
    gr.outputs.Image(type='file', label='Output'),
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=css,
    enable_queue=True
    ).launch(debug=True)