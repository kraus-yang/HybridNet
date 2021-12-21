import os
import numpy as np
import cv2
from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(rc={'figure.figsize':(50,50)})
palette = sns.color_palette("bright", 60)
from sklearn.manifold import TSNE

def get_data(feature_path,label_dir):
    all_feature=np.load((feature_path))
    with open(label_dir, 'rb') as f:
        _, label = pickle.load(f, encoding='latin1')
    # all_feature = all_feature[:1000]
    # label = label[:1000]
    n_samples, feature_dim = all_feature.shape
    #为前500个分配标签1，后500分配0
    return all_feature, label, n_samples, feature_dim

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    global fig
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i]),
    #              fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def plot_embedding_3D(data,label,title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    ax = plt.figure().add_subplot(111,projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9})
    return fig

#主函数
def main():
    feature_dir = '../work_dir/ntu/xview/agcn_feature_joint/features_test.npy'
    label_dir = './data/ntu/xview/val_label.pkl'
    print('Load data......')
    data, label, n_samples, n_features = get_data(feature_dir,label_dir) #根据自己的路径合理更改
    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    print('Finished......')
    np.save(os.path.split(feature_dir)[0]+'/t_sne_result.npy',result_2D)
    # result_2D = np.load('./work_dir/ntu/xview/agcn_feature_joint/t_sne_result.py.npy')
    ax=sns.scatterplot(result_2D[:, 0], result_2D[:, 1], hue=label, legend='full', palette=palette)

    #hide the grid and axes ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.setp(ax.get_legend().get_texts(), fontsize='4')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='4')  # for legend title

    num1 = 1.01
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(ncol=2,bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4, prop={'size': 10})

    # adjust legend_size
    # plt.legend(prop={'size': 10})
    # 2 column legend
    # plt.legend(ncol=2)
    # ax.legend_.remove()
    # plt.show(fig1)
    plt.savefig(os.path.split(feature_dir)[0]+'/visualization.jpg')
    plt.show()
    # fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    # plt.show(fig2)
if __name__ == '__main__':
    main()