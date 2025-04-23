import asyncio
import functools
import uuid

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr

LANG_CONFIG = {
    "ch": {"num_workers": 4},
    "en": {"num_workers": 4},
    "fr": {"num_workers": 1},
    "german": {"num_workers": 1},
    "korean": {"num_workers": 1},
    "japan": {"num_workers": 1},
}
CONCURRENCY_LIMIT = 8


class PaddleOCRModelWrapper(object):
    def __init__(self, model, name=None):
        super().__init__()
        self._model = model
        self._name = name or self._get_random_name()
        self._state = "IDLE"

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def infer(self, **kwargs):
        img_path = kwargs["img"]
        result = self._model.ocr(**kwargs)[0]
        image = Image.open(img_path).convert("RGB")
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores,
                        font_path="./simfang.ttf")
        return im_show

    def _get_random_name(self):
        return str(uuid.uuid4())


class PaddleOCRModelManager(object):
    def __init__(self,
                 num_models,
                 model_factory,
                 *,
                 polling_interval=0.1):
        super().__init__()
        self._num_models = num_models
        self._model_factory = model_factory
        self._polling_interval = polling_interval
        self._models = {}
        self.new_models()

    def new_models(self):
        self._models.clear()
        for _ in range(self._num_models):
            model = self._new_model()
            self._models[model.name] = model

    async def infer(self, **kwargs):
        while True:
            model = self._get_available_model()
            if not model:
                await asyncio.sleep(self._polling_interval)
                continue
            model.state = "RUNNING"
            # NOTE: I take an optimistic approach here, assuming that the model
            # is not broken even if inference fails.
            try:
                result = await self._new_inference_task(model, **kwargs)
            finally:
                model.state = "IDLE"
            return result

    def _new_model(self):
        real_model = self._model_factory()
        model = PaddleOCRModelWrapper(real_model)
        return model

    def _get_available_model(self):
        if not self._models:
            raise RuntimeError("No living models")
        for model in self._models.values():
            if model.state == "IDLE":
                return model
        return None

    def _new_inference_task(self, model,
                            **kwargs):
        return asyncio.get_running_loop().run_in_executor(
            None, functools.partial(model.infer, **kwargs))


def create_model(lang):
    return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False) 


model_managers = {}
for lang, config in LANG_CONFIG.items():
    model_manager = PaddleOCRModelManager(config["num_workers"], functools.partial(create_model, lang=lang))
    model_managers[lang] = model_manager


async def inference(img, lang):
    ocr = model_managers[lang]
    result = await ocr.infer(img=img, cls=True)
    return result


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
        gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='language')
    ],
    gr.Image(type='pil', label='Output'),
    title=title,
    description=description,
    examples=examples,
    cache_examples=False,
    css=css,
    concurrency_limit=CONCURRENCY_LIMIT,
    ).launch(debug=False)
