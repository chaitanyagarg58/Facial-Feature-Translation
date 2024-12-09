import numpy as np
import os

with open("data/celeba/list_attr_celeba.txt", "r") as file:
    all = file.readlines()
    header = all[:2]
    attr = all[2:]

folder = input("Data Folder to be generated: ")
num = int(input("Number of Images: "))

idx = np.random.choice(range(len(attr)), num, replace=False)

if os.path.exists(f"{folder}"):
    os.system(f"rm -r {folder}")

os.mkdir(f"{folder}")
os.mkdir(f"{folder}/images")
with open(f"{folder}/attr.txt", "w") as file:
    selected_attr = np.array(attr)[idx]
    splitStr = [elem.split() for elem in selected_attr]
    splitStr = [[f"{str(i+1).zfill(6)}.jpg", *(elem[1:]), "\n"] for i, elem in enumerate(splitStr)]
    selected_attr = [' '.join(elem) for elem in splitStr]
    file.writelines([f'{num}\n', header[1], *selected_attr])

for final, i in enumerate(idx):
    os.system(f"cp data/celeba/images/{str(i + 1).zfill(6)}.jpg {folder}/images/{str(final + 1).zfill(6)}.jpg")