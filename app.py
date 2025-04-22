from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr

lang_list = ['ch', 'en', 'fr', 'german', 'korean', 'japan']
ocr_dict = {lang: PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False) for lang in lang_list}

def inference(img, lang):
    ocr = ocr_dict[lang]

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
description = '''
- Gradio demo for PaddleOCR. PaddleOCR demo supports Chinese, English, French, German, Korean and Japanese. 
- To use it, simply upload your image and choose a language from the dropdown menu, or click one of the examples to load them. Read more at the links below.
- [Docs](https://paddlepaddle.github.io/PaddleOCR/), [Github Repository](https://github.com/PaddlePaddle/PaddleOCR).
'''

examples = [
    ['en_example.jpg','en'],
    ['cn_example.jpg','ch'],
    ['jp_example.jpg','japan'],
]

css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(
    inference,
    [
        gr.Image(type='filepath', label='Input'),
        gr.Dropdown(choices=lang_list, value='en', label='language')
    ],
    gr.Image(type='filepath', label='Output'),
    title=title,
    description=description,
    examples=examples,
    css=css
    ).launch(debug=False)
