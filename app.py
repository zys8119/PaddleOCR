from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import gradio as gr

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
def inference(img):
    img_path = img.name
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)
    
    # draw result
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    return 'result.jpg'

title = "PaddleOCR"
description = "Gradio demo for PaddleOCR. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

gr.Interface(
    inference, 
    [gr.inputs.Image(type="file", label="Input")], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article).launch(debug=True)