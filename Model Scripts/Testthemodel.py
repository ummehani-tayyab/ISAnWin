import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from PIL import Image
from misc import Utils
from collections import Counter
import matplotlib.pyplot as plt

root_dir = '..\..\..\Dataset\MVTEC_AD'
# model_path = r"..\models\model_at_epoch_10.pth" # old model
model_path = r"..\models\best_model_val_loss.pth" # updated model
directory = os.getcwd()

print('d= '+directory)
query_folder = r"..\data\query_folder"
target_folder = r"..\data\target_folder"
threshold = 0.3

def compare_images(query_path, target_path,target_folder,query_folder):
    query = Image.open(query_path)
    target = Image.open(target_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((240, 240)),
        transforms.ToTensor()])
    query = transform(query).unsqueeze(0)
    target = transform(target).unsqueeze(0)
    net = torch.load(model_path).cuda()
    with torch.no_grad():
        net.eval()
        output1, output2 = net(query.cuda(), target.cuda())
        distance = (F.pairwise_distance(output1, output2))
        concatenated = torch.cat((query, target), 0)
        #Utils.imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(distance.item())+target_folder+query_folder)
    return distance.item()


def test_all(query_folder, target_folder, threshold):
    query_categories = os.listdir(query_folder)
    target_categories = os.listdir(target_folder)
    num_categories = len(query_categories)
    confusion_matrix = np.zeros((num_categories, num_categories))
    num_true_positives = 0
    FP=[]
    FN=[]
    num_false_positives = 0
    num_true_negatives = 0
    num_false_negatives = 0
    for i, query_category in enumerate(query_categories):
        query_path = os.path.join(query_folder, query_category)
        query_files = os.listdir(query_path)
        for j, target_category in enumerate(target_categories):
            target_path = os.path.join(target_folder, target_category)
            target_files = os.listdir(target_path)

            for query_file in query_files:
                query_file_path = os.path.join(query_path, query_file)
                for target_file in target_files:
                    target_file_path = os.path.join(target_path, target_file)
                    difference = compare_images(query_file_path, target_file_path,target_category,query_category)
                    
                    if difference < threshold:
                        if query_category == target_category:
                            num_true_positives += 1
                            #print(f"{difference}, TP")
                            #switch:

                        else:
                            num_false_positives += 1 
                            print(query_file, target_file)
                            print(f"{difference}, FP")
                            FP.append(difference)

                    else:
                        if query_category == target_category:
                            num_false_negatives += 1
                            print(query_file, target_file)
                            print(f"{difference}, FN")
                            FN.append(difference)
                        else:
                            num_true_negatives += 1
                            #print(query_file, target_file)
                            #print(f"{difference}, TN")
    
    

    print(f"TP {num_true_positives}, TN {num_true_negatives}, FP {num_false_positives}, FN {num_false_negatives}")

    
    #print(f"FP {FP},FN {FN}")
    confusion_matrix[i, j] = (num_true_positives + num_true_negatives) / (num_true_positives + num_false_positives + num_true_negatives + num_false_negatives)
    rounded_array_FN = [round(value, 1) for value in FN]
    value_counts_FN = Counter(rounded_array_FN)
    most_common_value_FN = value_counts_FN.most_common(1)[0][0]
    most_common_count_FN = value_counts_FN.most_common(1)[0][1]
    print(f"Most Common FN: {most_common_value_FN}, Most Common value Count:, {most_common_count_FN} ")

    rounded_array_FP = [round(value, 1) for value in FP]
    value_counts_FP = Counter(rounded_array_FP)
    #most_common_value_FP = value_counts_FP.most_common(1)[0][0]
    #most_common_count_FP = value_counts_FP.most_common(1)[0][1]
    #print(f"Most Common FP: {most_common_value_FP}, Most Common value Count:, {most_common_count_FP} ")
                
    return confusion_matrix, query_categories, target_categories ,num_true_positives, num_false_positives, num_true_negatives, num_false_negatives

"""confusion_matrix, query_categories, target_categories = test_all(query_folder, target_folder, threshold)
print("Confusion matrix:")
print(" ".join(["{:>10}".format(cat[:5]) for cat in target_categories]))
for i in range(len(query_categories)):
    print("{:<5}".format(query_categories[i][:5]), end="")
    for j in range(len(target_categories)):
        print("{:>10.2f}".format(confusion_matrix[i, j]), end="")
    print()
    """

confusion_matrix, query_categories, target_categories,num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = test_all(query_folder, target_folder, threshold)
def acc():
    accuracies = []
    total_queries = num_false_positives+num_true_negatives+num_false_negatives+num_true_positives
    total_matches = num_true_positives + num_true_negatives
    accuracy = (total_matches * 100) / total_queries 
    accuracies.append(accuracy)
    return accuracies

print("Confusion matrix:")
print(" ".join(["{:>10}".format(cat[:5]) for cat in target_categories]))
for i in range(len(query_categories)):
    print("{:<5}".format(query_categories[i][:5]), end="")
    for j in range(len(target_categories)):
        print("{:>10.2f}".format(confusion_matrix[i, j]), end="")
    print(" TP:{:<3} TN:{:<3} FP:{:<3} FN:{:<3}".format(
        int(confusion_matrix[i, j] * (num_true_positives + num_false_negatives)),
        int(confusion_matrix[i, j] * (num_true_negatives + num_false_positives)),
        int((1 - confusion_matrix[i, j]) * num_false_positives),
        int((1 - confusion_matrix[i, j]) * num_false_negatives)
    )) 

accuracy = acc()
plt.plot(accuracy)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.show()
