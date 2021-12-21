import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xview', choices={'kinetics', 'ntu/xsub', 'ntu/xview','ntu120/xsetup','ntu120/xsub'},
                    help='the work folder for storing results')
arg = parser.parse_args()

dataset = arg.datasets
label = open('/home/kraus/PycharmProjects/RS-GCN/data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
ind = np.argsort(label[0])
label = label[:,ind]
r1 = open('./work_dir/' + dataset + '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r1 =sorted(r1,key=lambda x: x[0])
right_num = np.zeros(len(set(label[1])))
total_num = np.zeros(len(set(label[1])))
for i in tqdm(range(len(label[0]))):
    file1, l = label[:, i]
    file2, r = r1[i]
    r = np.argmax(r)
    l=int(l)
    if l == r:
        right_num[l] += 1
    total_num[l] += 1
class_acc = right_num / total_num
class_acc = {str(i):class_acc[i] for i in range(len(class_acc))}
pickle.dump(class_acc, open('../class_acc_hybrid_net.pkl', 'wb'))
print(class_acc)