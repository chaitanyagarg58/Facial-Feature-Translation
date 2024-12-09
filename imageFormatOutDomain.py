from PIL import Image
import os
import argparse
from torchvision import transforms as T

parser = argparse.ArgumentParser()

parser.add_argument("--gan", type=str, default="stargan")

gan = parser.parse_args().gan

selected_attrs = ["Blond_Hair", "Bald", "Smiling", "Young"]
if gan == "attgan":
    selected_attrs = ["Blond_Hair", "Bald", "Young"]


output_size = (128 * (len(selected_attrs) + 1), 128 * 8)  
result_image = Image.new("RGB", output_size)


for i in range(1, 9):
    real_img = Image.open(f"outDomainData/images/{i}.jpg")
    result_image.paste(real_img.resize((128,128)), (0, (i - 1) * 128))

    for idx, attr in enumerate(selected_attrs):
        img = Image.open(f"results/outDomainData/{gan}/{attr}/{i}_{attr}.jpg")

        result_image.paste(img, ((idx + 1) * 128, (i - 1) * 128))



result_image.save(f"{gan}.jpg")

result_image.show()
