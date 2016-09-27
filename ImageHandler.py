'''
Created on Sep 5, 2016

@author: Denalex
'''
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from astropy.io.votable.validator.result import Result

class ImageHandler:
    
    def __init__(self, path_name):
        self.path_name = path_name
        self.files = [f for f in listdir(self.path_name) if isfile(join(self.path_name, f))]
        
    def setPath(self, path_name):
        self.path_name = path_name
        return
    
    def update_files(self):
        self.files = [f for f in listdir(self.path_name) if isfile(join(self.path_name, f))]
    
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
    def rgb_to_grey_weighted(self, image_name):
        im = Image.open(self.path_name + image_name)
        im = im.convert('L')
        return im
    
    @staticmethod
    def get_noise_from_greyscale(self, image_name, image_width, image_height):
        im = Image.open(self.path_name + image_name)
        im = im.convert('L')
        pixelArray = np.array(im)
        multiplier = 1 / (36 * (image_width - 2) * (image_height - 2))
        sum = 0 
        return multiplier * sum
    
    @staticmethod
    def get_segmented_noise(self, image_name):
        return
    
    def get_neural_input(self, classifier = 0, file_name = ""):
        return