# This file will use to score your implementations.
# You should not change this file

import os
import pandas as pd
import time
import sys
import cv2

from Module import predict

if __name__ == "__main__":

    input_folder = sys.argv[1]
    label_file = sys.argv[2]

    df_labels = pd.read_csv(label_file)
    img_name = df_labels['img_name'].values.tolist()
    img_labels = df_labels['label'].values.tolist()
    img_to_label = dict([(img_name[i], img_labels[i]) for i in range(len(img_name))])

    start_time = time.time()
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))
    fail_process = 0
    cnt_predict = 0

    start_time = time.time()
    for filename in list_files:
        img_path = os.path.join(input_folder, filename)
        print(cv2.imread(img_path).shape)
        try:
            label = predict(img_path)
        except:
            label = -1

        if img_to_label[filename] == label:
            cnt_predict += 1
        elif label == -1:
            print(filename)
            fail_process += 1

    run_time = time.time() - start_time

    print("Ket qua dung: %i/%i" % (cnt_predict, len(list_files)))
    print("Loi: %i" % fail_process)
    print("Score = %.2f" % (10.*cnt_predict/len(list_files)))
    print("Run time in: %.2f s" % run_time)
