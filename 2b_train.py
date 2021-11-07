import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import configparser
import argparse
import os
from utils import print_config
import matplotlib.pyplot as plt

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data





def find_center_in_line(line):
    if not 'images_original' in line:
        return line.split(' ')[0].split('/')[-1].split('_')[4], line.split(' ')[0].split('/')[-1].split('_')[5]
    else:
        return line.split(' ')[0].split('/')[-1].split('_')[6], line.split(' ')[0].split('/')[-1].split('_')[7]





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





def shuffle_samples(lines, annotation_path):
    
    file_annotations = open(annotation_path,'w')

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    
    file_annotations.writelines(lines)





def _main(annotation_path, log_dir, classes_path, anchors_path, model_path, input_shape_height, input_shape_width, patients_train, patients_val, patients_test, train_frozen_network, train_unfrozen_network, batch_size_freeze, batch_size_unfreeze, initial_epoch_freeze, last_epoch_freeze, initial_epoch_unfreeze, last_epoch_unfreeze, learning_rate_frozen_network, learning_rate_unfrozen_network, number_of_scale):
    
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (input_shape_height,input_shape_width)

    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path)

    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=2)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2)

    with open(annotation_path) as f:
        lines = f.readlines()
        
    shuffle_samples(lines, annotation_path)
        
    train, val, test = build_train_val_test_list(lines, number_of_scale, patients_train, patients_val, patients_test)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if train_frozen_network:
        model.compile(optimizer=Adam(lr=learning_rate_frozen_network), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=["acc"])

        batch_size = batch_size_freeze
        history = model.fit_generator(data_generator_wrapper(train, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, len(train)//batch_size),
                validation_data=data_generator_wrapper(val, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, len(val)//batch_size),
                shuffle=True,
                epochs=last_epoch_freeze,
                initial_epoch=initial_epoch_freeze,
                callbacks=[logging])
        model.save_weights(log_dir + '_frozen_network.h5')
        
        # summarize history for loss
        history_loss_frozen_network = history.history['loss']
        history_val_loss_frozen_network = history.history['val_loss']



    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if train_unfrozen_network:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=learning_rate_unfrozen_network), loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=["acc"]) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = batch_size_unfreeze # note that more GPU memory is required after unfreezing the body
        history = model.fit_generator(data_generator_wrapper(train, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, len(train)//batch_size),
            validation_data=data_generator_wrapper(val, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, len(val)//batch_size),
            shuffle=True,
            epochs=last_epoch_unfreeze,
            initial_epoch=initial_epoch_unfreeze,
            callbacks=[logging, reduce_lr, early_stopping])
        model.save_weights(log_dir + '_unfrozen_network.h5')

        # summarize history for loss
        history_loss_unfrozen_network = history.history['loss']
        history_val_loss_unfrozen_network = history.history['val_loss']
        
        history_loss = history_loss_frozen_network + history_loss_unfrozen_network
        history_val_loss = history_val_loss_frozen_network + history_val_loss_unfrozen_network
        
        plt.plot(history_loss)
        plt.plot(history_val_loss)
        plt.ylim(0, 200)
        plt.grid()
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path_logs + '/network_loss.png')
        plt.clf()
       	plt.cla()



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)



def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model



def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model



def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)
        
        

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)





if __name__ == '__main__':
    # Create the file arguments
    parser = argparse.ArgumentParser(description='Train the COCO pre-trained model with glomeruli dataset.')
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

    augmented_dataset_name = config.get('Paths', 'augmented_dataset_name')
    path_dataset_created = config.get('Paths', 'path_dataset_created')
    annotations_file_name = config.get('Paths', 'annotations_file_name')
    path_logs = config.get('Paths', 'path_logs')
    model_path = config.get('Paths', 'model_path')
    classes_path = config.get('Paths', 'classes_path')
    anchors_path = config.get('Paths', 'anchors_path')
    final_annotations_file_name = config.get('Paths', 'final_annotations_file_name')

    
    dataset_name = config.get('Train', 'dataset_name')
    input_shape_height = config.getint('Train', 'input_shape_height')
    input_shape_width = config.getint('Train', 'input_shape_width')
    patients_train = config.get('Train', 'patients_train')
    patients_val = config.get('Train', 'patients_val')
    patients_test = config.get('Train', 'patients_test')
    train_frozen_network = config.getboolean('Train', 'train_frozen_network')
    train_unfrozen_network = config.getboolean('Train', 'train_unfrozen_network')
    batch_size_freeze = config.getint('Train', 'batch_size_frozen_network')
    batch_size_unfreeze = config.getint('Train', 'batch_size_unfrozen_network')
    initial_epoch_freeze = config.getint('Train', 'initial_epoch_freeze')
    last_epoch_freeze = config.getint('Train', 'last_epoch_freeze')
    initial_epoch_unfreeze = config.getint('Train', 'initial_epoch_unfreeze')
    last_epoch_unfreeze = config.getint('Train', 'last_epoch_unfreeze')
    learning_rate_frozen_network = config.getfloat('Train', 'learning_rate_frozen_network')
    learning_rate_unfrozen_network = config.getfloat('Train', 'learning_rate_unfrozen_network')

    number_of_scale = len(config.get('MultiScale', 'scale_list').split(','))

    annotation_path = path_dataset_created + dataset_name + '/' + final_annotations_file_name

    if display_logs :
        print_config(config_path)
    
    _main(annotation_path, path_logs, classes_path, anchors_path, model_path, input_shape_height, input_shape_width, patients_train, patients_val, patients_test, train_frozen_network, train_unfrozen_network, batch_size_freeze, batch_size_unfreeze, initial_epoch_freeze, last_epoch_freeze, initial_epoch_unfreeze, last_epoch_unfreeze, learning_rate_frozen_network, learning_rate_unfrozen_network, number_of_scale)
