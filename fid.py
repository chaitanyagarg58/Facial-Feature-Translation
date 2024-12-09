import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import scipy.linalg
from PIL import Image
import os


# Load InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def calculate_activation_statistics(images):
    # Preprocess images for Inceptionv3 model
    images = preprocess_input(images)

    # Get intermediate layer activations
    activations = model.predict(images)

    # Calculate mean and covariance matrix of activations
    print(images.size, activations.size)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma

def calculate_frechet_inception_score(real_images, fake_images, num_samples=1000):
    # Calculate activation statistics for real and fake images
    mu_real, sigma_real = calculate_activation_statistics(real_images)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images)

    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake, disp = False)
    score = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2.0*covmean)

    return np.real(score)

nm = len(os.listdir("testData/images"))
real_images = np.array(Image.open('testData/images/000001.jpg').resize((128,128)))[np.newaxis,:,:,:]

for i in range(2, nm + 1):
	x = str(i).zfill(6)
	r = np.array(Image.open(f'testData/images/{x}.jpg').resize((128,128)))[np.newaxis,:,:,:]
	real_images = np.concatenate([real_images, r], axis = 0)

real_images = real_images[1:,:,:,:]

attributes = ["Blond_Hair", "Bald", "Young"]
gans = ["stargan", "attentiongan", "attgan"]

for gan in gans:
	for attr in attributes:
		fake_images = np.array(Image.open(f'results/testData/{gan}/{attr}/1_{attr}.jpg').resize((128,128)))[np.newaxis,:,:,:]
		for i in range(2, nm + 1):
			r = np.array(Image.open(f'results/testData/{gan}/{attr}/{i}_{attr}.jpg'))[np.newaxis,:,:,:]
			fake_images = np.concatenate([fake_images, r], axis = 0)

		fake_images = fake_images[1:,:,:,:]

		fid_score = calculate_frechet_inception_score(real_images, fake_images, nm)

		with open(f"fid_score_{gan}.txt", 'a') as f:
			f.write(f"{attr}: {fid_score}\n")
