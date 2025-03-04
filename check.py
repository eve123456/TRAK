import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def get_descriptive_stats(arr, name):
    df = pd.DataFrame({name: arr})
    summary = df.describe()
    print(summary)


def view_histplot(data,bins, x_label,  title, save_dir):
    plt.hist(data, bins)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(save_dir)
    plt.clf()

def symmetric_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def plot_top_bottom(indices, flag, tag, save_dir):
    # flag == "positive" or "negative"
    # tag == "0", "1", ..., "9", "Overall" 
    plt.figure(figsize = (15,10))
    sub_idx = 1
    for idx_trn in indices:
        score = str(np.round(overall_score[idx_trn],8))
        plt.subplot(2, 5, sub_idx)
        plt.imshow(train_set.data[idx_trn])
        plt.axis("off")
        plt.title(f"Training Sample ID {idx_trn}\nLabel: {train_set.targets[idx_trn]} {label_dict[train_set.targets[idx_trn]]}\nScore: "+score, fontsize=14)
        sub_idx += 1
    plt.suptitle("Top 10 " + flag + " influential pairs on Class -- " + tag, fontsize=16)
    plt.tight_layout(pad=4.0, w_pad=0.1, h_pad=2.0)
    plt.savefig(save_dir)



def plot_pairs(train_indices, target_indices, save_dir):
    l = len(train_indices)
    plt.figure(figsize=(7,40))
    sub_idx = 1
    for idx_trn, idx_tgt in zip(train_indices, target_indices):
        score = str(np.round(scores[idx_trn,idx_tgt],3))
        print(score)
        plt.subplot(l, 2, sub_idx)
        plt.imshow(train_set.data[idx_trn])
        plt.axis("off")
        plt.title(f"Training Sample ID {idx_trn}\nLabel: {train_set.targets[idx_trn]} {label_dict[train_set.targets[idx_trn]]}\nScore: "+score, fontsize=14)
        sub_idx += 1
        plt.subplot(l, 2, sub_idx)
        plt.imshow(test_set.data[idx_tgt])
        plt.axis("off")
        plt.title(f"Target Sample ID {idx_tgt}\nLabel: {test_set.targets[idx_tgt]} {label_dict[test_set.targets[idx_tgt]]}\nScore: "+score,fontsize=14)
        sub_idx += 1
    plt.suptitle("Top 10 positive influential pairs", fontsize=16)
    plt.tight_layout(pad=4.0, w_pad=0.1, h_pad=2.0)
    plt.savefig(save_dir)    


def plot_extreme_for_class(indices, save_dir):
    plt.figure(figsize=(9,40))
    sub_idx = 1   
    for i_class in range(10):
        top, bottom = int(indices[i_class,0]), int(indices[i_class,1])
        plt.subplot(10, 2, sub_idx)
        plt.imshow(train_set.data[top])
        plt.axis("off")
        plt.title(f"On Class {i_class} {label_dict[i_class]} Most Positive\n Training Sample ID {top}\nLabel: {train_set.targets[top]} {label_dict[train_set.targets[top]]}", fontsize=14)
        sub_idx += 1
        plt.subplot(10, 2, sub_idx)
        plt.imshow(train_set.data[bottom])
        plt.axis("off")
        plt.title(f"On Class {i_class} {label_dict[i_class]} Most Negative\n Training Sample ID {bottom}\nLabel: {train_set.targets[bottom]} {label_dict[train_set.targets[bottom]]}", fontsize=14)
        sub_idx += 1
    plt.suptitle("Most Influential Samples for Each Class", fontsize=16)
    plt.tight_layout(pad=4.0, w_pad=1.0, h_pad=2.0)
    plt.savefig(save_dir)   


label_dict = {0:"airplane", 
              1: "automobile",
              2: "bird",
              3: "cat",
              4: "deer",
              5: "dog",
              6: "frog",
              7: "horse",
              8: "ship",
              9: "truck"}


scores = np.load("./trak_results/scores/scores_0129.npy")
print(scores.shape) # 50,000 x 10,000

train_set = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                        download=True,
                                        train=True)

test_set = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                        download=True,
                                        train=False)

##########################################
#### influence: sample on sample
##########################################
# single_score = scores.reshape(-1)
# # obtain descripive statistics
# get_descriptive_stats(arr = single_score,
#                       name = "Sample to Sample Score")
# # view the histplot
# view_histplot( data = symmetric_log(single_score),
#                 bins = 100,
#                 x_label= "TRAK influence score (Symmetric log)",
#                 title = "Histplot: sample-to-sample influence score: \n ResNet-9 on CIFAR-10",
#                 save_dir = "./sample_to_sample_histplot.jpg")
                
# top 10 positive influential pairs
# top_10_indices = np.unravel_index(np.argsort(scores, axis=None)[-10:], scores.shape)
# print(top_10_indices)
# posi_trn_indices, posi_tgt_indices = (np.array([18685,  4934, 36449, 43730, 36449, 36449, 26629, 43730, 36449,
#         26629]), np.array([9661, 8983, 8352,  467, 1608, 8934, 9625, 5281, 1297, 5665]))
# plot_pairs(train_indices= posi_trn_indices, 
#            target_indices = posi_tgt_indices, 
#            save_dir = "./top_10_positive_pairs.jpg")

# top 10 negative influential pairs
# lowest_10_indices = np.unravel_index(np.argsort(scores, axis=None)[:10], scores.shape)
# print(lowest_10_indices)
# posi_trn_indices, posi_tgt_indices = (np.array([36449, 36449, 18685, 36449, 36449, 43730, 36449, 43730, 43730,
#        43730]), np.array([1694, 9819, 3280, 8960, 9826, 1125, 2241, 3453, 9761,  905]))
# plot_pairs(train_indices= posi_trn_indices, 
#            target_indices = posi_tgt_indices, 
#            save_dir = "./top_10_negative_pairs.jpg")

##########################################
#### influence: sample on overall performance
##########################################
# # calculate influence of all training examples on the overall performance
# overall_score = np.sum(scores,axis = 1)/len(scores[0])

# # # obtain descripive statistics
# # get_descriptive_stats(arr = overall_score, name = "overall score")

# # # view the histplot
# # view_histplot(data = symmetric_log(overall_score),
# #                 bins = 100,
# #                 x_label= "TRAK influence score (Symmetric log)",
# #                 title = "Histplot: Influence score of all training samples on overall performance: \n ResNet-9 on CIFAR-10",
# #                 save_dir = "./overall_histplot.jpg")


# # top_10_indices = np.argsort(overall_score)[-10:][::-1]
# # print(top_10_indices)
# top_10_indices = [49313, 38311, 4987, 6894, 35690, 23510, 34634, 29685, 36288, 34203]
# # bottom_10_indices = np.argsort(overall_score)[:10][::-1]
# # print(bottom_10_indices)
# bottom_10_indices = [3536, 3851, 20518, 38332, 13805, 4934, 10380, 36449, 18685, 28438]

# plot_top_bottom(top_10_indices, "positive", "overall", "./top_10_positive_overall.jpg")
# plot_top_bottom(bottom_10_indices, "negative", "overall", "./top_10_negative_overall.jpg")


# top 10 negative influential pairs
# lowest_10_indices = np.unravel_index(np.argsort(scores, axis=None)[:10], scores.shape)
# print(lowest_10_indices)
# posi_trn_indices, posi_tgt_indices = (np.array([36449, 36449, 18685, 36449, 36449, 43730, 36449, 43730, 43730,
#        43730]), np.array([1694, 9819, 3280, 8960, 9826, 1125, 2241, 3453, 9761,  905]))
# plot_pairs(train_indices= posi_trn_indices, 
#            target_indices = posi_tgt_indices, 
#            save_dir = "./top_10_negative_pairs.jpg")


# calculate influence of all training examples on class i 

# classes_score = np.zeros((50000,10))
# for i_class in tqdm(range(10)):
#     indices = np.array(test_set.targets) == i_class
#     for i in range(50000):
#         classes_score[i][i_class] = sum(scores[i][indices])/sum(indices)
    
    

# np.save("./score_to_class.npy", np.array(classes_score))



# classes_score = np.load("./score_to_class.npy")
# most_influence_idx = np.zeros((10,2))

# for i_class in range(9):
#     sorted_influence = np.argsort(classes_score[:,i_class])
#     top, bottom = sorted_influence[-1], sorted_influence[0]
#     most_influence_idx[i_class, 0] = top
#     most_influence_idx[i_class, 1] = bottom
    
# np.save("./class_influential_index.npy", most_influence_idx)
most_influence_idx = np.load("./class_influential_index.npy")
plot_extreme_for_class(most_influence_idx, "./most_influential_for_class.jpg")




# train_set.data[0] if of 3x32x32
# train_set.targets[0] is the label (0-9)









