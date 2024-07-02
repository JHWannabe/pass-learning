import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
import cv2
import shutil
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def confusion_matrix_pytorch(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes)

    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1

    return cm

def save_to_csv(roc_auc, average_precision, confusion_matrix, epoch, file_path):
    # Calculate additional metrics
    true_positives = confusion_matrix.diag().tolist()
    false_negatives = [(confusion_matrix[i].sum().item() - tp) for i, tp in enumerate(true_positives)]
    false_positives = [(confusion_matrix[:, i].sum().item() - tp) for i, tp in enumerate(true_positives)]
    true_negatives = [(confusion_matrix.sum().item() - tp - fn - fp) for tp, fn, fp in zip(true_positives, false_negatives, false_positives)]

    correct_predictions = confusion_matrix.diag().sum().item()
    total_predictions = confusion_matrix.sum().item()

    accuracy = correct_predictions / total_predictions

    accuracy = [round(accuracy * 100, 1)]
    roc_auc = [round(roc_auc, 4)]
    average_precision = [round(average_precision, 4)]

    # Prepare data for CSV
    data = zip([epoch], true_positives, true_negatives, false_negatives, false_positives, accuracy, roc_auc, average_precision)

    # Write data to CSV file
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Check if file is empty to write header
        if file.tell() == 0:
            writer.writerow(['Epoch', 'NG to NG', 'GOOD to GOOD', 'NG to GOOD', 'GOOD to NG', 'Accuracy', 'ROC', 'Average Precision'])
        writer.writerows(data)
        
    return accuracy

def test_only(supervised_model, dataloader, folder_path, num, target, device: str = 'cpu'):
    supervised_model.eval()
    y_true_list = []
    y_pred_list = []
    prob_list = []

    with torch.no_grad():
        count = 0
        good_path = f'{folder_path}/{num}/GOOD'
        ng_path = f'{folder_path}/{num}/NG'

        for idx, (inputs, masks, y_true, file_name) in enumerate(dataloader):
            inputs, masks = inputs.to(device), masks.to(device)
            y_true_list.append(y_true.item())

            zero = np.zeros(inputs.shape)

            # supervised predict
            outputs_ = supervised_model(inputs)
            outputs = F.sigmoid(outputs_)
            final_outputs = outputs
            final_segmap = outputs.cpu().detach().numpy()
            final_segmap = final_segmap.squeeze()

            # save weight with real data for torchscript
            supervised_model.eval()

            img = cv2.cvtColor(inputs.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_RGB2GRAY)
            img = np.uint8(min_max_norm(img) * 255)

            ## combined final segmap
            s_threshold = 0.1
            final_segmap = np.where(final_segmap > s_threshold, 1, 0)
            combined_final_segmap = np.where(final_segmap > 0 , 255, 0)     

            if np.all(combined_final_segmap == zero):
                y_pred_list.append(1)
                probs = torch.max(1-final_outputs)
                isExist = os.path.exists(good_path)
                if not isExist :
                    os.makedirs(good_path)
                cv2.imwrite(f'{good_path}/{str(file_name[0])}.jpg', img)
            else:
                y_pred_list.append(0)
                probs = torch.max(final_outputs)
                isExist = os.path.exists(ng_path)
                if not isExist :
                    os.makedirs(ng_path)
                cv2.imwrite(f'{ng_path}/{str(file_name[0])}.jpg', img)
                cv2.imwrite(f'{ng_path}/{str(file_name[0])}_segmap.jpg', combined_final_segmap)

            prob_list.append(probs)

            count = count + 1

        # 실제 값과 예측 값을 numpy 배열로 변환
        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)
        prob_array = torch.tensor(prob_list)

        # ROC 곡선 계산
        fpr, tpr, thresholds = roc_curve(y_true_array, prob_array)
        roc_auc = auc(fpr, tpr)

        # Average Precision 계산
        average_precision = average_precision_score(y_true_array, prob_array)

        good_exists = os.path.exists(good_path)
        ng_exists = os.path.exists(ng_path)

        if ng_exists and good_exists:
            cm = confusion_matrix_pytorch(torch.tensor([y_true_list]), torch.tensor([y_pred_list]), 2)
            save_to_csv(roc_auc, average_precision, cm, num, f'{folder_path}/{target}_confusion_metrics.csv')
        else:
            cm = confusion_matrix_pytorch(torch.tensor([y_true_list]), torch.tensor([y_pred_list]), 2)
            save_to_csv(roc_auc, average_precision, cm, num, f'{folder_path}/{target}_confusion_metrics.csv')
            shutil.rmtree(f'{folder_path}/{num}')

        # supervised_model.eval()
        # tt = torch.jit.trace(supervised_model, inputs).to('cuda:0')
        # tt.save('D:/JHChun/model_weight/zumul/before/zumul_supervised.pth')

        