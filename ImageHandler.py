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
    
    def __init__(self, path_name, save_path_name):
        self.path_name = path_name
        self.save_path_name = save_path_name
        self.files = [f for f in listdir(self.path_name) if isfile(join(self.path_name, f))]
        
    def setPath(self, path_name):
        self.path_name = path_name
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
        im = Image.open(self.path_name + image_name)
        im = im.convert('L')
        if (save):
            im.save(self.save_path_name + image_name)
        return im
    
    @staticmethod
    def get_noise_from_greyscale(self, image_name, image_width, image_height):
        im = Image.open(self.path_name + image_name)
        im = im.convert('L')
        pixelArray = np.array(im)
        mask = np.matrix([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
        multiplier = 1.0 / (36.0 * (image_width - 2.0) * (image_height - 2.0))
        sum = 0.0
        for i in range(0, image_width):
            for j in range(0, image_height):
                new_mask = mask * pixelArray[i, j]
                sum = sum + (new_mask.sum() ** 2)
        return multiplier * sum
    
    #reads noise from parts of an image
    @staticmethod
    def get_segmented_noise_28(self, image_name):
        return
    
    def get_neural_input(self, classifier = 0, file_name = ""):
        return