from utils.helper_functions.src_folder_path import src_folder_path


training_data_folder_path = '/data/'
name = 'msg_train.csv'
training_data_set = src_folder_path+training_data_folder_path+name

testing_dataset_name = 'msg_test.csv'
testing_data_set = src_folder_path+training_data_folder_path+testing_dataset_name

final_output_dataset_name = 'final_output_dataset.csv'
final_output_dataset = src_folder_path+training_data_folder_path+final_output_dataset_name

Id_key='Id'
Camera_key='Camera'
Camera_key_old='Camera_old'
Angle_key='Angle'
Width_key='Width'
Height_key='Height'
Facing_key='Facing\r'
Facing_key_old='Facing_old'
final_column_key = 'final_column_key'


#Model Properties [start]
#learning_rate = 0.01
#epochs = 2
#batch_size = 32
#embed_dim = 50
#max_len = 10
#seed = 7
#Model Properties [end]

#Model Properties [start]
learning_rate = 0.01
epochs = 100
batch_size = 16
embed_dim = 500
max_len = 4
seed = 7
#Model Properties [end]