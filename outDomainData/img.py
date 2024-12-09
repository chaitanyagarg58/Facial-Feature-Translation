from PIL import Image

name = input("Image Name: ")
size = 178, 218
im = Image.open("./images/" + name + ".jpg")
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("./images/" + name + ".jpg")