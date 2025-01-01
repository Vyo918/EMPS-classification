# # split dataset to train(80%) and test(20%)

# ## DON'T RUN, OR ELSE GONNA SPLIT AGAIN


# import os
# import shutil
# import random

# src_dir = "./dataset/EMPS-datasets/model1/"
# train_dir = "./dataset/my_dataset/model1/train/"
# test_dir = "./dataset/my_dataset/model1/test/"

# # num = 2000
# split_ratio = 0.8
    
# for category in os.listdir(src_dir):
    
#     label_dir = os.path.join(src_dir, category)
#     if os.path.isdir(label_dir):
#         train_label_dir = os.path.join(train_dir, category)
#         test_label_dir = os.path.join(test_dir, category)
#         os.makedirs(train_label_dir, exist_ok=True)
#         os.makedirs(test_label_dir, exist_ok=True)
        
#         files = os.listdir(label_dir)
#         files = [f for f in files if os.path.isfile(os.path.join(label_dir, f))]
#         # del files[num:] # decrease the dataset into 1.5k per category
#         print(len(files))
        
#         random.shuffle(files)
        
#         split_index = int(len(files) * split_ratio)
        
#         train_files = files[:split_index]
#         test_files = files[split_index:]
        
#         for file in train_files:
#             src_path = os.path.join(label_dir, file)
#             dest_path = os.path.join(train_label_dir, file)
#             # print(src_path, dest_path)
#             shutil.copy2(src_path, dest_path)
        
#         for file in test_files:
#             src_path = os.path.join(label_dir, file)
#             dest_path = os.path.join(test_label_dir, file)
#             # print(src_path, dest_path)
#             shutil.copy2(src_path, dest_path)
            
#         print(f"Processed label '{category}': {len(train_files)} training files, {len(test_files)} testing files.")