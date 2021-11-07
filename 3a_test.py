import configparser
import argparse
import os
from utils import print_config





def build_train_val_test_list(lines, number_of_scale, patients_train, patients_val, patients_test):
    train, val, test = [], [], []
    
    for line in lines:
        subline = line.split('/')[-1]

        if subline[7:9] in str(patients_train):
            train.append(line)
        if subline[7:9] in str(patients_val):
            val.append(line)
        if subline[7:9] in str(patients_test):
            test.append(line)

        
    print('[TRAIN] Number of images : ', len(train), ' / split size : ', round(len(train) / (len(train) + len(val) + len(test)), 2) * 100, '%')
    print('[VAL]   Number of images : ', len(val), ' / split size : ', round(len(val) / (len(train) + len(val) + len(test)), 2) * 100, '%')
    print('[TEST]  Number of images : ', len(test), ' / split size : ', round(len(test) / (len(train) + len(val) + len(test)), 2) * 100, '%')
                 

    # Remove images in test produced by data augmentation
    test = [line for line in test if 'IFTA' in line.split(' ')[0].split('/')[-1][0:4]]

    return train, val, test





if __name__ == '__main__':
    # Create the file arguments
    parser = argparse.ArgumentParser(description='Create a file fill with detections performed by the network.')
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

    path_dataset_created = config.get('Paths', 'path_dataset_created')
    path_logs = config.get('Paths', 'path_logs')
    model_path = config.get('Paths', 'model_path')
    final_annotations_file_name = config.get('Paths', 'final_annotations_file_name')

    dataset_name = config.get('Train', 'dataset_name')
    patients_train = config.get('Train', 'patients_train')
    patients_val = config.get('Train', 'patients_val')
    patients_test = config.get('Train', 'patients_test')
    
    annotation_path = path_dataset_created + dataset_name + '/' + final_annotations_file_name

    number_of_scale = len(config.get('MultiScale', 'scale_list').split(','))

    if display_logs :
        print_config(config_path)
        
            
    with open(annotation_path) as f_gt:
        lines_gt = f_gt.readlines()
    
    _, _, test = build_train_val_test_list(lines_gt, number_of_scale, patients_train, patients_val, patients_test)

 
    for line in test:
        image_path = line.split(' ')[0]
        os.system('python yolo_image.py --input ' + image_path + ' --model model_data/yolo.h5 >> ' + path_logs + '/annotations_results.txt')



    f_gt.close()
