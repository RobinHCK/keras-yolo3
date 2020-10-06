import shutil
import os
import sys
import cv2
from imutils import paths
import random
import configparser
import argparse
from utils import print_config, buildAnnotationsFile
from imantics import Mask
from PIL import Image
from PIL import ImageEnhance





def create_dataset_hierarchy(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name):
    # Create the hierarchy dataset
    
    try:  
        if not os.path.exists(path_dataset_created):
            os.mkdir(path_dataset_created)
        
        if os.path.exists(path_dataset_created + augmented_dataset_name):
            shutil.rmtree(path_dataset_created + augmented_dataset_name)
        os.mkdir(path_dataset_created + augmented_dataset_name)
        os.mkdir(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name)
        os.mkdir(path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name)
    except OSError:  
        print ('Creation of the directory failed')
        
    if display_logs :
        print('Dataset hierarchy created', path_dataset_created + augmented_dataset_name)
        
        

        

def fillAnnotationsFile(path_dataset_created, augmented_dataset_name, dataset_masks_name, annotations_file_name, type_of_border) :
    # Open masks newly created by data augmentation and calculate de box or mask to fill the annotations file
    
    file_annotations = open(path_dataset_created + augmented_dataset_name + '/' + annotations_file_name,'a')
    
    paths_masks = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name))
    
    for path in paths_masks :
         
        mask = cv2.imread(path,0)
                            
        polygon = Mask(mask).polygons().points
        
        # The type of annotation is a mask
        if type_of_border == 'mask':
            str_polygon = '['
                    
            for poly in polygon:
                str_polygon += '['
                for points in poly:
                    str_polygon = str_polygon + '[' + str(points[0]) + ',' + str(points[1]) + '],'
                str_polygon = str_polygon[:-1]
                str_polygon += '],'
            str_polygon = str_polygon[:-1] + ']'
        # The type of annotation is a box        
        elif type_of_border == 'box':   
            str_polygon = '['
                 
            minX = sys.maxsize
            minY = sys.maxsize
            maxX = -sys.maxsize
            maxY = -sys.maxsize
        
            for poly in polygon:
                for point in poly:
                    if point[0] < minX:
                        minX = point[0]
                    if point[1] < minY:
                        minY = point[1]
                    if point[0] > maxX:
                        maxX = point[0]   
                    if point[1] > maxY:
                        maxY = point[1] 

                str_polygon += '[' + str(minX) + ',' + str(minY) + ',' + str(maxX) + ',' + str(maxY) + ',0],'

                minX = sys.maxsize
                minY = sys.maxsize
                maxX = -sys.maxsize
                maxY = -sys.maxsize
                
            str_polygon = str_polygon[:-1] + ']'
        # Wrong value for the type of annotation
        else :
            raise ValueError('The type of border given does not exist (location : config.cfg - Annotations - type_of_border) : ', type_of_border)
        
        mask_name = path.split('/')[-1]
        file_annotations.write(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name + '/' + mask_name + ' ' + str_polygon + '\n')

    file_annotations.close()
    
    if display_logs :
        print('Fill annotations file with newly generated masks')


   
    
   
def transfer_original_images(path_dataset_created, augmented_dataset_name, dataset_name, dataset_images_name, dataset_masks_name):
    paths_images = list(paths.list_images(path_dataset_created + dataset_name + '/' + dataset_images_name))
    paths_masks = list(paths.list_images(path_dataset_created + dataset_name + '/' + dataset_masks_name))
    
    for path in paths_images :
        image_name = str(path.split('/')[-1])
        
        image = cv2.imread(path,1)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_images_name + '/' + image_name), image)

    for path in paths_masks :
        mask_name = str(path.split('/')[-1])
        
        mask = cv2.imread(path,0)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name + '/' + mask_name), mask)
    
    
    
    
    
# =============================================================================
# Data Augmentation Method  
# =============================================================================
def data_augmentation_flip(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name):
    paths_images = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name))
    paths_masks = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name))
    
    for path in paths_images :
        direction = random.randint(0,1)
        image_name = "flip_" + str(path.split('/')[-1])
                
        image = cv2.imread(path,1)
        image = cv2.flip(image, direction)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_images_name + '/' + image_name), image)

    for path in paths_masks :
        direction = random.randint(0,1)
        mask_name = "flip_" + str(path.split('/')[-1])
        
        mask = cv2.imread(path,0)
        mask = cv2.flip(mask, direction)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name + '/' + mask_name), mask)
    




def data_augmentation_contrast(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name):
    paths_images = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name))
    paths_masks = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name))
    
    for path in paths_images :
        contrast = random.randint(90,110) / 100.0
        image_name = "contrast_" + str(path.split('/')[-1])
                
        image = Image.open(path)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        image.save(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name + '/' + image_name)

    for path in paths_masks :
        mask_name = "contrast_" + str(path.split('/')[-1])
        
        mask = cv2.imread(path,0)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name + '/' + mask_name), mask)
    
    
    
    
    
def data_augmentation_rotate(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name, patch_size_height, patch_size_width):
    paths_images = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_images_name))
    paths_masks = list(paths.list_images(path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name))
    
    patch_size = (patch_size_width, patch_size_height)
    patch_center = (patch_size_width//2, patch_size_height//2)
    
    for path in paths_images :
        angle = random.randint(1,3)*90
        image_name = "rotate_" + str(path.split('/')[-1])
                
        image = cv2.imread(path,1)
        rotation = cv2.getRotationMatrix2D(patch_center, angle, 1.0)
        image = cv2.warpAffine(image, rotation, patch_size)
        image = image[patch_center[1]//2:patch_center[1]+patch_center[1]//2, patch_center[0]//2:patch_center[0]+patch_center[0]//2]
        image = cv2.resize(image,patch_size)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_images_name + '/' + image_name), image)

    for path in paths_masks :
        angle = random.randint(1,3)*90
        mask_name = "rotate_" + str(path.split('/')[-1])
        
        mask = cv2.imread(path,0)
        rotation = cv2.getRotationMatrix2D(patch_center, angle, 1.0)
        mask = cv2.warpAffine(mask, rotation, patch_size)
        mask = mask[patch_center[1]//2:patch_center[1]+patch_center[1]//2, patch_center[0]//2:patch_center[0]+patch_center[0]//2]
        mask = cv2.resize(mask,patch_size)
        cv2.imwrite((path_dataset_created + augmented_dataset_name + '/' + dataset_masks_name + '/' + mask_name), mask)
    


    
    
# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    
    # Create the file arguments
    parser = argparse.ArgumentParser(description='Perform a data augmentation on ARGD dataset.')
    parser.add_argument('-c', '--config', type=str, help='The configuration file to use')
    args = parser.parse_args()
    
    # Read the config file given in parameter, if not read the default config file
    if args.config:
        config_path = args.config
    else:
        config_path = 'config.cfg'
        
    if not os.path.isfile(config_path):
        raise ValueError('The configuration file path given does not exist : ', config_path)
        
    config = configparser.RawConfigParser(interpolation = configparser.ExtendedInterpolation())
    config.read(config_path)
    
    
    
    # Get the usefull informations from config file
    display_logs = config.getboolean('General', 'display_logs')

    dataset_name = config.get('Paths', 'dataset_name')
    augmented_dataset_name = config.get('Paths', 'augmented_dataset_name')
    path_dataset_created = config.get('Paths', 'path_dataset_created')
    dataset_images_name = config.get('Paths', 'dataset_images_name')
    dataset_masks_name = config.get('Paths', 'dataset_masks_name')
    annotations_file_name = config.get('Paths', 'annotations_file_name')
    final_annotations_file_name = config.get('Paths', 'final_annotations_file_name')

    type_of_border = config.get('Annotations', 'type_of_border')
    
    patch_size_height = config.getint('MultiScale', 'patch_size_height')
    patch_size_width = config.getint('MultiScale', 'patch_size_width')
    
    flip = config.getboolean('DataAugmentation', 'flip')
    rotate = config.getboolean('DataAugmentation', 'rotate')
    contrast = config.getboolean('DataAugmentation', 'contrast')

    if display_logs :
        print_config(config_path)


    
    # Build the new dataset
    create_dataset_hierarchy(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name)
    
    # Transfer original images
    transfer_original_images(path_dataset_created, augmented_dataset_name, dataset_name, dataset_images_name, dataset_masks_name)
    
    # Load the transformations
    if flip :
        data_augmentation_flip(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name)
        
    if contrast :
        data_augmentation_contrast(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name)

    if rotate :
        data_augmentation_rotate(path_dataset_created, augmented_dataset_name, dataset_images_name, dataset_masks_name, patch_size_height, patch_size_width)
        
    if display_logs :
        print('Data augmentation successfully finished')
        
        
    
    # Add new annotations generated by data augmentation to annotations file
    fillAnnotationsFile(path_dataset_created, augmented_dataset_name, dataset_masks_name, annotations_file_name, type_of_border)
    
    buildAnnotationsFile(path_dataset_created, augmented_dataset_name, annotations_file_name, final_annotations_file_name, display_logs)
    
    
