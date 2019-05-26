import pandas as pd
from properties.properties import *

final_input_column = list()


def data_to_sequence(data):
    for index, row in data.iterrows():
        final_input_column.append(data_from_row(row))

    return final_input_column


def data_from_row(row):

    return [int(row[Camera_key]),
            row[Angle_key],
            row[Height_key],
            row[Width_key]]


def clean_and_store_data(data_set):
    data = retrieve_data(data_set)

    data[Camera_key_old] = data[Camera_key]

    data[Camera_key] = convert_facing_numeric(data[Camera_key])

    try:
        data[Facing_key_old] = data[Facing_key]
        data[Facing_key] = convert_facing_numeric(data[Facing_key])
    except:
        pass
    return data


def convert_facing_numeric(facing):
    return facing.apply(
            lambda x: '1' if (x == 'F\r' or x == 'F')
            else '2' if (x == 'L\r' or x == 'L')
            else '3' if (x == 'R\r' or x == 'R')
            else '4')


def convert_numeric_to_facing(x):
    if x == 0 or x == '0':
        return 'F'
    elif x == 1 or x == '1':
        return 'L'
    elif x == 2 or x == '2':
        return 'R'
    else:
        return 'X'


def retrieve_data(data_set):
    data = pd.read_csv(data_set, encoding="ISO-8859-1", lineterminator='\n')
    data = data.copy()
    return data
