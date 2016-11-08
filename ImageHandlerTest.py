'''
Created on Sep 29, 2016

@author: Denalex
'''
from ImageHandler import ImageHandler as IH

if __name__ == '__main__':
    handler = IH("C:\\Users\\Denalex\\Desktop\\ImageHandlerTest\\Read", "C:\\Users\\Denalex\\Desktop\\ImageHandlerTest\\Save")
    print(handler.files)
    dataset = handler.get_neural_input(classifier=1)
    print("\nTest Dataset (Noise Values): \n" + str(dataset))