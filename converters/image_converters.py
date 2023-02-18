import sys
from converters.text_converters import GoogleTextConverter
from logger.Logger import Logger

class LavisImageToVectorConverter:

    def __init__(self, text_converter):
        self.text_converter = text_converter
        sys.path.insert(0, 'LAVIS')
        from lavis.models import load_model_and_preprocess
        self.device = 'cpu'
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)

    def convert(self, img):
        image = self.vis_processors['eval'](img).unsqueeze(0).to(self.device)
        captions = self.model.generate({'image': image})
        Logger.d("img2caption", captions[0])
        return self.text_converter.convert(captions[0])
