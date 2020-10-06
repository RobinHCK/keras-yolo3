# Print the entire config file content
def print_config(config_path) :
    print('config.cfg :\n')
    print(open(config_path, 'r').read())
    
    
    
def buildAnnotationsFile(path_dataset_created, dataset_name, annotations_file_name, final_annotations_file_name, display_logs) :
    file_annotations = open(path_dataset_created + dataset_name + '/' + annotations_file_name,'r+')
    new_file_annotations = open(path_dataset_created + dataset_name + '/' + final_annotations_file_name,'w+')

    new_content = file_annotations.read().replace('],[', ' ').replace(']', '').replace('[', '')

    new_file_annotations.write(new_content)
    
    file_annotations.close()
    new_file_annotations.close()
    
    if display_logs :
            print('Annotations file ready to use')