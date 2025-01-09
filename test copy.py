import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
import cv2
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score

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
            writer.writerow(['Epoch', ' NG to NG', ' GOOD to GOOD', ' NG to GOOD', ' GOOD to NG', ' Accuracy', ' ROC', ' Average Precision'])
        writer.writerows(data)
        
    return accuracy


def merge_and_delete_images(directory, base_name, segmap):
    image_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith('.jpg')]

    # 이미지를 읽고 리스트에 저장합니다
    images = [cv2.imread(os.path.join(directory, img)) for img in image_files]
    
    # 이미지들의 높이를 동일하게 맞춥니다
    min_height = min(img.shape[0] for img in images)
    images = [cv2.resize(img, (img.shape[1], min_height)) for img in images]
    
    # 이미지들을 가로로 연결합니다
    merged_image = np.hstack(images)
    
    # 병합된 이미지를 저장합니다
    if segmap:
        cv2.imwrite(os.path.join(os.path.dirname(directory), f"{base_name}_segmap.jpg"), merged_image)
    else:
        cv2.imwrite(os.path.join(os.path.dirname(directory), f"{base_name}.jpg"), merged_image)
    
    # 원본 이미지들을 삭제합니다
    for img_file in image_files:
        os.remove(os.path.join(directory, img_file))

    return merged_image


def test_only(supervised_model, dataloader, folder_path, num, target, device: str = 'cpu'):
    supervised_model.eval()
    y_true_list = []
    y_pred_list = []
    prob_list = []

    with torch.no_grad():
        count = 0
        temp_path = f'{folder_path}/{num}/Temp'
        good_path = f'{folder_path}/{num}/GOOD'
        ng_path = f'{folder_path}/{num}/NG'
        overkill_path = f'{folder_path}/{num}/Overkill'
        notfound_path = f'{folder_path}/{num}/Notfound'

        paths = [temp_path, good_path, ng_path, overkill_path, notfound_path]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        if not os.path.exists(os.path.join(temp_path, 'origin')):
                os.makedirs(os.path.join(temp_path, 'origin'))
        if not os.path.exists(os.path.join(temp_path, 'segmap')):
            os.makedirs(os.path.join(temp_path, 'segmap'))

        # tqdm을 사용하여 진행 상황 표시
        for idx, (inputs, masks, y_true, file_name, inputs_org) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[{num}] Testing"):
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
            inputs_org = np.uint8(inputs_org).squeeze()

            ## combined final segmap
            s_threshold = 0.1
            final_segmap = np.where(final_segmap > s_threshold, 1, 0)
            combined_final_segmap = np.where(final_segmap > 0 , 255, 0)

            if target == 'land':
                cv2.imwrite(f'{temp_path}/origin/{str(file_name[0])}.jpg', inputs_org)
                cv2.imwrite(f'{temp_path}/segmap/{str(file_name[0])}_segmap.jpg', combined_final_segmap)

                image_files = [f for f in os.listdir(f'{temp_path}/origin') if f.startswith(str(file_name[0]).split('_')[0]) and f.endswith('.jpg')]
                if len(image_files) % 4 == 0:
                    combined_final_input = merge_and_delete_images(f'{temp_path}/origin', str(file_name[0]).split('_')[0], False)
                    combined_final_segmap = merge_and_delete_images(f'{temp_path}/segmap', str(file_name[0]).split('_')[0], True)

                    if np.all(combined_final_segmap == 0):
                        y_pred_list.append(1)
                        probs = torch.max(final_outputs)
                        if y_true == 0:
                            new_path = notfound_path
                        elif y_true == 1:
                            new_path = good_path
                    else:
                        y_pred_list.append(0)
                        probs = torch.max(1-final_outputs)
                        if y_true == 0:
                            new_path = ng_path
                        elif y_true == 1:
                            new_path = overkill_path

                    shutil.move(f'{temp_path}/{str(file_name[0]).split("_")[0]}.jpg', f'{new_path}/{str(file_name[0]).split("_")[0]}.jpg')
                    shutil.move(f'{temp_path}/{str(file_name[0]).split("_")[0]}_segmap.jpg', f'{new_path}/{str(file_name[0]).split("_")[0]}_segmap.jpg')
                else:
                    probs = torch.tensor(0.0)

                prob_list.append(probs)


            else:
                if np.all(combined_final_segmap == zero):
                    y_pred_list.append(1)
                    probs = torch.max(1-final_outputs)
                    if y_true == 0:
                        cv2.imwrite(f'{notfound_path}/{str(file_name[0])}.jpg', img)
                    # elif y_true == 1:
                    #     cv2.imwrite(f'{good_path}/{str(file_name[0])}.jpg', inputs_org)
                else:
                    y_pred_list.append(0)
                    probs = torch.max(final_outputs)
                    if y_true == 0:
                        cv2.imwrite(f'{ng_path}/{str(file_name[0])}.jpg', img)
                        cv2.imwrite(f'{ng_path}/origin/{str(file_name[0])}.jpg', inputs_org)
                        cv2.imwrite(f'{ng_path}/{str(file_name[0])}_segmap.jpg', combined_final_segmap)
                    elif y_true == 1:
                        cv2.imwrite(f'{overkill_path}/{str(file_name[0])}.jpg', img)
                        cv2.imwrite(f'{overkill_path}/origin/{str(file_name[0])}.jpg', inputs_org)
                        cv2.imwrite(f'{overkill_path}/{str(file_name[0])}_segmap.jpg', combined_final_segmap)

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

        cm = confusion_matrix_pytorch(torch.tensor([y_true_list]), torch.tensor([y_pred_list]), 2)
        save_to_csv(roc_auc, average_precision, cm, num, f'{folder_path}/{target}_confusion_metrics.csv')