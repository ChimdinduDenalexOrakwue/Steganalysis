'''
Created on Sep 5, 2016

@author: Denalex
'''
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from astropy.io.votable.validator.result import Result
from pygame import PixelArray

class ImageHandler:
    
    def __init__(self, path_name, save_path_name):
        self.path_name = str(path_name)
        self.save_path_name = save_path_name
        self.files = [f for f in listdir(self.path_name) if isfile(join(self.path_name, f))]
        
    def setPath(self, path_name):
        self.path_name = path_name
        return
    
    def setSavePath(self, save_path_name):
        self.save_path_name = save_path_name
        return
    
    def update_files(self):
        self.files = [f for f in listdir(self.path_name) if isfile(join(self.path_name, f))]
        
    def print_files(self):
        for i in range(0, len(self.files)):
            print(self.files[i])
        return
    
    @staticmethod
    def rgb_to_grey_average(self, image_name):
        im = Image.open(self.path_name + image_name)
        pixelArray = np.array(im)
        result = np.array()
        for i in range(0, len(pixelArray)):
            sum = 0
            for j in range(0, len(pixelArray[i])):
                sum += pixelArray[i][j]
            result += [sum/3]
                
        return result
    
    @staticmethod
    def rgb_to_grey_weighted(self, image_name, save = False):
        im = Image.open(str(self.path_name + image_name))
        im = im.convert('L')
        if (save):
            im.save(self.save_path_name + image_name)
        return im
    
    def get_noise_from_greyscale_noisemap(self, image_name = "", image_width = 28, image_height = 28):
        im = Image.open(image_name)
        im = im.convert('L')
        pixelArray = np.array(im)
        noiseMap = np.zeros((28, 28))
        multiplier = 1.0 / (36.0 * (image_width - 2.0) * (image_height - 2.0))
        sum = 0.0
        for i in range(0, image_width):
            sum = 0.0
            for j in range(0, image_height):
                val = self.apply_mask(i, j, pixelArray)
                noiseMap[i, j] = val
                sum = sum + val
        print("sum = " + str(sum))
        return noiseMap
    
    def apply_mask(self, x, y, image):
        maskIndex = 0
        sum = 0
        mask = [1.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 1.0]
        for row in range(x - 1, x + 2):
            for column in range(y - 1, y + 2):
                if self.in_bounds(row, column, image):
                    sum = sum + (image[row, column] * mask[maskIndex])
                    maskIndex = maskIndex + 1
        return (sum / 9)
    
    def in_bounds(self, x, y, image):
        return (x >= 0 and x < len(image)) and (y >=0 and y<len(image))
        
        
    #reads noise from parts of an image
    @staticmethod
    def get_segmented_noise_28(self, image_name):
        return
    
    def get_neural_input(self, classifier = 0):
        dataset = np.empty(shape=[0,1])
        self.update_files()
        for i in range(0, len(self.files)):
            noise = self.get_noise_from_greyscale(image_name = self.path_name + "\\" + self.files[i])
            dataset = np.append(dataset, [[noise]], axis = 0)
        return dataset
    
    def get_neural_input_noisemap(self, classifier = 1):
        dataset = np.empty(shape=[28,28])
        self.update_files()
        for i in range(0, len(self.files)):
            noise = self.get_noise_from_greyscale_noisemap(image_name = self.path_name + "\\" + self.files[i])
            dataset = np.append(dataset, noise, axis = 0)
        return dataset
        