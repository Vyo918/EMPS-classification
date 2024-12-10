#split dataset to train(80%) and test(20%)

### DON'T RUN, OR ELSE GONNA SPLIT AGAIN


# import os
# import shutil
# import random

# src_dir = "C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/dataset/EMPS-datasets/classification_model#1_dataset/train/"
# train_dir = "C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/dataset/my_dataset/model1/train/"
# test_dir = "C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/dataset/my_dataset/model1/test/"

# split_ratio = 0.8

# for big_category in os.listdir(src_dir):
#     big_label_dir = os.path.join(src_dir, big_category)
    
#     for small_category in os.listdir(big_label_dir):
        
#         small_label_dir = os.path.join(big_label_dir, small_category)
#         if os.path.isdir(small_label_dir):
#             train_label_dir = os.path.join(train_dir, big_category, small_category)
#             test_label_dir = os.path.join(test_dir, big_category, small_category)
#             os.makedirs(train_label_dir, exist_ok=True)
#             os.makedirs(test_label_dir, exist_ok=True)
            
#             files = os.listdir(small_label_dir)
#             files = [f for f in files if os.path.isfile(os.path.join(small_label_dir, f))]
            
#             random.shuffle(files)
            
#             split_index = int(len(files) * split_ratio)
            
#             train_files = files[:split_index]
#             test_files = files[split_index:]
            
#             for file in train_files:
#                 src_path = os.path.join(small_label_dir, file)
#                 dest_path = os.path.join(train_label_dir, file)
#                 print(src_path, dest_path)
#                 # shutil.copy2(src_path, dest_path)
            
#             for file in test_files:
#                 src_path = os.path.join(small_label_dir, file)
#                 dest_path = os.path.join(test_label_dir, file)
#                 print(src_path, dest_path)
#                 # shutil.copy2(src_path, dest_path)
                
#             print(f"Processed label '{small_category}': {len(train_files)} training files, {len(test_files)} testing files.")
            

            