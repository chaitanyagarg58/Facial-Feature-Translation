from PIL import Image
import os
import argparse
from torchvision import transforms as T

parser = argparse.ArgumentParser()

selected_attrs = ["Blond_Hair", "Bald", "Smiling", "Young"]

att_selected_attrs = ["Blond_Hair", "Bald", "Young"]

gans = ["stargan", "attentiongan", "attgan"]

for i in range(4):
    output_size = (128 * 5, 128 * 3)  
    result_image = Image.new("RGB", output_size)


    real_img = Image.open(f"testData/images/{str(i+1).zfill(6)}.jpg")

    for idx, gan in enumerate(gans):
        result_image.paste(real_img.resize((128,128)), (0, (idx) * 128))
        
        if idx < 2:
            for j, attr in enumerate(selected_attrs):
                
                img = Image.open(f"results/testData/{gan}/{attr}/{i + 1}_{attr}.jpg")

                result_image.paste(img, ((j + 1) * 128, (idx) * 128))
        else:
            for j, attr in enumerate(att_selected_attrs):
                
                img = Image.open(f"results/testData/{gan}/{attr}/{i + 1}_{attr}.jpg")
                if (j == 2):
                    result_image.paste(img, ((j + 2) * 128, (idx) * 128))
                else:
                    result_image.paste(img, ((j + 1) * 128, (idx) * 128))


    result_image.save(f"{i+1}.jpg")

    result_image.show()
