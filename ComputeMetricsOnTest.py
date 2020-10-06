import numpy as np
import configparser
import argparse
import os
import time
import cv2
from utils import print_config
import sys
import xlsxwriter





# =============================================================================
# FUNCTIONS
# =============================================================================
def findGTLineFromDetectionLine(line_detections, lines_gts):
    for line_gts in lines_gts :
        if line_detections.split(' ')[0].split('/')[-1] == line_gts.split(' ')[0].split('/')[-1] :
                return line_gts





def drawRectangleInImage(x1, y1, x2, y2, img, color, th):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    




def convertDetections(detections):
    return detections.replace('\'','').replace('"','').replace('[','').replace(']','').replace(' ','').replace('\n','').split('glomeruli')[1:]

    

    

def result_is_a_correct_detection(gt, detection, threshold_iou):
    if len(gt) == 0: 
         return False    
     
    # create table filled with 0 (no dectection), 1 (predicted detection), 2 (groundtruth detection) and 3 (1 & 2)
    result_pixels = np.array([[0 for col in range(1500)] for row in range(1500)])
    
    for x in range(int(detection[0]), int(detection[2])):
        for y in range(int(detection[1]), int(detection[3])):
            result_pixels[x, y] = 1
    
    gt_pixels = np.array([[0 for col in range(1500)] for row in range(1500)])
    
    for x in range(int(gt[0]), int(gt[2])):
        for y in range(int(gt[1]), int(gt[3])):
            gt_pixels[x, y] = 2
                
    merge_pixels = result_pixels + gt_pixels
    
    # Compute the shared area between the groundtruth and predicted detections
    number_of_pixels_detected_by_prediction = np.count_nonzero(merge_pixels == 1)
    number_of_pixels_detected_by_gt = np.count_nonzero(merge_pixels == 2)
    number_of_pixels_detected_by_gt_and_prediction = np.count_nonzero(merge_pixels == 3)
    
    shared_area = number_of_pixels_detected_by_gt_and_prediction / (number_of_pixels_detected_by_prediction + number_of_pixels_detected_by_gt + number_of_pixels_detected_by_gt_and_prediction)

    if shared_area > threshold_iou:
        return True
        
    return False





def build_confusion_matrice_for_one_image(gt, result, threshold_iou):
    TP = []
    FP = []
    FN = []

    # formating data
    formating_result = []
    result = result.split('.png ')[1:]
    for detection in result[0].replace('[[', '').replace(']]', '').split('], ['):
        positions = detection.split(', ')
        positions.append(positions.pop(0))
        
        formating_result.append(positions)
    
    formating_gt = []

    gt = gt.split('.png ')[1:]

    for detection in gt[0].split(' '):
        positions = detection.split(',')
        positions.pop()
        
        formating_gt.append(positions)
    
    # classify detections
    # filter with confidence score
    new_formating_result = []
    for result in formating_result:
        if float(result[-1].split(' ')[-1][:-1]) >= confidence:
            new_formating_result.append(result)
            
    formating_result = new_formating_result
    # filter with IOU
    for gt in formating_gt:
        detection_is_correct = False

        for result in formating_result:
            if result_is_a_correct_detection(gt, result, threshold_iou):
                detection_is_correct = True
                TP.append(result)

        if not detection_is_correct:
            FN.append(gt)
    # If a detection is not correct (TP), the detection is FP.
    for r in formating_result:
        if r not in TP:
            FP.append(r)
    
    return TP, FP, FN





# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and write metrics in datas.xlsx')
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
    
    
    display_logs = config.getboolean('General', 'display_logs')
    path_logs = config.get('Paths', 'path_logs')

    patients_test = config.get('Train', 'patients_test')
    path_images = config.get('Paths', 'path_images') 
    lod = config.getint('MultiScale', 'lod')
    scale_list = config.get('MultiScale', 'scale_list')

    path_dataset_created = config.get('Paths', 'path_dataset_created')
    dataset_name = config.get('Train', 'dataset_name')
    annotations_file_name = config.get('Paths', 'final_annotations_file_name')
    annotation_path = path_dataset_created + dataset_name + '/' + annotations_file_name
    
    IOU = config.get('Eval', 'IOU')
    confidenceScore = config.get('Eval', 'confidenceScore')

    tmp = []
    for e in scale_list.split(','):
        tmp.append(int(e))
    scale_list = tmp
    
    tmp = []
    for e in IOU.split(','):
        tmp.append(float(e))
    IOU = tmp
    
    tmp = []
    for e in confidenceScore.split(','):
        tmp.append(float(e))
    confidenceScore = tmp
    
    
    
    sys.stdout = open(path_logs + '/log_ComputeMetricsOnTest.txt', 'w')
    
    if display_logs :
        print_config(config_path)
    
    # Open annotations file
    with open(annotation_path) as f_gts:
        lines_gts = f_gts.readlines()
        
        
        
    # Open results file
    annotations_results_file_name = path_logs + '/annotations_results.txt'
    
    with open(annotations_results_file_name) as f_detections:
        lines_detections = f_detections.readlines()
        
    

    wrapper_lines_detections = []
    for s in scale_list:
        tmp = []
        for line_detections in lines_detections :
            if ('scale_' + str(s)) in line_detections.split(' ')[0].split('/')[-1] :
                tmp.append(line_detections)
        wrapper_lines_detections.append(tmp)
    
    
    
    for lines_detections in wrapper_lines_detections:
        scale = int(lines_detections[0].split(' ')[0].split('/')[-1].split('_')[-1][:-4])
                    
        start = time.time()    
        
        results = []
        
        for iou in IOU:
            for confidence in confidenceScore:
                
                print('\nEvaluation in progress :\tIOU', iou,'\tConfidence', confidence)
                
                for patient in patients_test.split(',') :
                    print('Patient :', patient, '- Scale :', scale)
                    
                    tp_count = 0
                    fp_count = 0
                    fn_count = 0
                                
                    for line_detections in lines_detections :
                        TP = []
                        FP = []
                        FN = []
                        
                        if ('IFTA_00' + patient) in line_detections.split(' ')[0].split('/')[-1] :
                            line_gts = findGTLineFromDetectionLine(line_detections, lines_gts)
                            
                            detections = convertDetections(line_detections)
                            gts = line_gts.split(' ')[1:]
                            
                            patch_x = int(line_detections.split(' ')[0].split('/')[-1].split('_')[-5]) - int(100/(lod*lod))
                            patch_y = int(line_detections.split(' ')[0].split('/')[-1].split('_')[-3]) - int(100/(lod*lod))
                                
                            TP, FP, FN = build_confusion_matrice_for_one_image(line_gts, line_detections, iou)

                        tp_count += len(TP)
                        fp_count += len(FP)
                        fn_count += len(FN)
                    
                    if (tp_count+fp_count) == 0:
                        precision = 0
                    else :
                        precision = tp_count / (tp_count+fp_count)
                        
                    if (tp_count+fn_count) == 0:
                        rappel = 0
                    else :
                        rappel = tp_count / (tp_count+fn_count)
                        
                    if (precision+rappel) == 0:
                        F1score = 0
                    else :
                        F1score = (2*precision*rappel) / (precision+rappel)
                    
                    print('TP\t\t', tp_count, '\nFP\t\t', fp_count, '\nFN\t\t', fn_count)
                    print('precision\t', round(precision,3), '\nrappel\t\t', round(rappel,3))
                    print('F1 Score\t', round(F1score,3))
    
                    results.append([iou, confidence, tp_count, fp_count, fn_count, precision, rappel, F1score, patient])
    
        end = time.time()
        print('Execution time :', round(((end-start)/60), 2), 'min')
        
        # Save datas in Excel file
        workbook = xlsxwriter.Workbook(path_logs + '/datas_' + str(scale) + '.xlsx')
        worksheet = workbook.add_worksheet() 
        
        row = 1
        column = 0
        
        worksheet.write(0, 0, 'iou')
        worksheet.write(0, 1, 'confidence')
        worksheet.write(0, 2, 'TP')
        worksheet.write(0, 3, 'FP')
        worksheet.write(0, 4, 'FN')
        worksheet.write(0, 5, 'precision')
        worksheet.write(0, 6, 'rappel')
        worksheet.write(0, 7, 'F1score')
        worksheet.write(0, 8, 'patient')
        
        
        for result in results:
            column = 0
            for e in result :
                worksheet.write(row, column, e)
                column += 1
            row += 1
            
        workbook.close()

    
