from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import os
from tqdm import tqdm
import utils
import argparse


class VITGPT2_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        gpu = ",".join([str(i) for i in utils.get_free_gpu()[0 : 1]])
        utils.set_gpu(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model.eval()

        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(self, image_paths):
        images = []
        image_names = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            image_names.append(os.path.split(image_path)[1])
            images.append(i_image)
        
        with torch.no_grad():
            pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.cuda()

            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{image_names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(self, img_folder, csv_path, path_pos=0, label_pos=1, batch_size=256):
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        lines = [x.strip() for x in open(csv_path, "r").readlines()][1:]
        save_path = os.path.join(img_folder, "vit-gpt2_captions.csv")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                caption_lines = f.readlines()
            if len(caption_lines) == len(lines):
                print(f"{save_path} have been generated")
                return save_path
        count = 0
        timer = utils.Timer()
        with open(save_path, "w") as fout:
            while count < len(lines):
                sel = lines[count:min(count+batch_size,len(lines))]
                paths = [os.path.join(img_folder, s.split(',')[path_pos]) for s in sel]
                labels = [s.split(',')[label_pos].strip() for s in sel]
                msgs = self.predict_step(paths)
                write_info = '\n'.join([f"{msgs[i]},{labels[i]}" for i in range(len(msgs))])
                fout.write(f"{write_info}\n")
                fout.flush()
                count += batch_size
                elapsed_time = timer.t()
                est_time = elapsed_time/count * len(lines)
                print(f"Progress: {count/len(lines)*100:.2f}% {utils.time_str(elapsed_time)}/est:{utils.time_str(est_time)}   ", end="\r")
        return save_path
