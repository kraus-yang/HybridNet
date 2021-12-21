import pickle
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_class(file1, file2, class_name, save_dir='./picture'):
    c_acc1 = pickle.load(open(file1, 'rb'))
    c_acc2 = pickle.load(open(file2, 'rb'))
    model_name1 = file1.split('/')[-1][10:-4]
    model_name2 = file2.split('/')[-1][10:-4]
    plt.figure(dpi=300, figsize=(30, 10))
    df = pd.DataFrame(columns=('class', 'acc', 'model'))
    for key, value in c_acc1.items():
        df = df.append([{'class': class_name[int(key)], 'acc': value, 'model': model_name1}])
        df = df.append([{'class': class_name[int(key)], 'acc': c_acc2[key], 'model': model_name2}])
    g = sns.barplot(x="class", y="acc", hue="model", data=df)
    g.legend_.set_title('')
    # sns.barplot(x="data_size", y="acc", data=df,palette=color_list)

    plt.ylim([0.65, 1.00])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc=2, fontsize=12)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(save_dir + '/' + 'class_accuracy_of.eps')
    # plt.show()


if __name__ == '__main__':
    classname = ['drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up', 'throw', 'sit down',
                 'stand up', 'clapping', 'reading', 'writing', 'tear up paper', 'put on jacket', 'take off jacket',
                 'put on a shoe','take off a shoe', 'put on glasses', 'take off glasses', 'put on a hat/cap',
                 'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket',
                 'hopping', 'jump up', 'phone call',  'play with phone/tablet', 'type on a keyboard',
                 'point to something', 'taking a selfie', 'check time','rub two hands', 'nod head/bow',
                 'shake head', 'wipe face', 'salute', 'put palms together','cross hands in front', 'sneeze/cough',
                 'staggering','falling down','headache', 'chest pain',  'back pain', 'neck pain',  'nausea/vomiting',
                 'fan self', 'punch/slap', 'kicking', 'pushing', 'pat on back', 'point finger','hugging',
                 'giving object', 'touch pocket', 'shaking hands', 'walking towards', 'walking apart']
    file1 = '../class_acc_agcn.pkl'
    file2 = '../class_acc_hybrid_net.pkl'
    plot_class(file1, file2, classname)
