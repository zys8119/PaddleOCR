import os
os.system('pip install paddlepaddle')
os.system('pip install paddleocr')
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False)

def inference(img):
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
description = 'Gradio demo for PaddleOCR. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'
article = "<p style='text-align: center'><a href='https://www.paddlepaddle.org.cn/hub/scene/ocr'>Awesome multilingual OCR toolkits based on PaddlePaddle （practical ultra lightweight OCR system, support 80+ languages recognition, provide data annotation and synthesis tools, support training and deployment among server, mobile, embedded and IoT devices）</a> | <a href='https://github.com/PaddlePaddle/PaddleOCR'>Github Repo</a></p>"

gr.Interface(
    inference,
    gr.inputs.Image(type='file', label='Input'),
    gr.outputs.Image(type='file', label='Output'),
    title=title,
    description=description,
    article=article,
    ).launch(debug=True)