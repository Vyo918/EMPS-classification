import csv
import os
import shutil

with open('Classification_Model_2/dataset/FashionStyle14/train.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        old_path = 'Classification_Model_2/dataset/FashionStyle14/' + row[0]
        new_path = 'Classification_Model_2/dataset/FashionStyle14/data_csv/train/' + '/'.join(row[0].split('/')[1:])
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
        except FileNotFoundError:
            print(f"File not found: {old_path}")
            
with open('Classification_Model_2/dataset/FashionStyle14/test.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        old_path = 'Classification_Model_2/dataset/FashionStyle14/' + row[0]
        new_path = 'Classification_Model_2/dataset/FashionStyle14/data_csv/test/' + '/'.join(row[0].split('/')[1:])
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
        except FileNotFoundError:
            print(f"File not found: {old_path}")
            
with open('Classification_Model_2/dataset/FashionStyle14/val.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        old_path = 'Classification_Model_2/dataset/FashionStyle14/' + row[0]
        new_path = 'Classification_Model_2/dataset/FashionStyle14/data_csv/val/' + '/'.join(row[0].split('/')[1:])
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
        except FileNotFoundError:
            print(f"File not found: {old_path}")