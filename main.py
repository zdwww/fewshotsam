import cv2
import numpy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
from utils.utils import *
import time
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def get_embedding(img, predictor):
    predictor.set_image(img)
    img_emb = predictor.get_image_embedding()
    return img_emb


def train(args, predictor, model_choie, name):
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    fnames = [f for f in os.listdir(os.path.join(data_path, 'images', 'train')) if f.lower().endswith(('png', 'jpg'))]
    print(f"Train frames: {fnames}")
    num_image = len(fnames)

    image_embeddings = []
    labels = []
    
    # get the image embeddings
    print('Start training...')
    t1 = time.time()
    for fname in tqdm(fnames):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', 'train', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'annotation', 'train', fname.replace('jpg', 'png')))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY) # threshold the mask to 0 and 1
        downsampled_mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
         
        img_emb = get_embedding(image, predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        image_embeddings.append(img_emb)

        labels.append(downsampled_mask.flatten())
        
    t2 = time.time()
    print("Time used: {}m {}s".format((t2 - t1) // 60, (t2 - t1) % 60))
    image_embeddings_cat = np.concatenate(image_embeddings)
    labels = np.concatenate(labels)

    # Create a linear regression model and fit it to the training data
    #model = LogisticRegression(max_iter=1000) 
    print("model name:", name)
    model_choie.fit(image_embeddings_cat, labels)
    
    return model_choie

def test_visualize(args, model, predictor):
    data_path = args.data_path
        
    fnames = [f for f in os.listdir(os.path.join(data_path, 'images', 'test')) if f.lower().endswith(('png', 'jpg'))]
    print(f"Test frames: {len(fnames)}")
    num_image = len(fnames)
    num_visualize = args.visualize_num
    
    dice_linear = []
    dice1 = []
    dice2 = []
    dice3 = []
    f1_score3 = []
    iou_scores = []
    f1_scores = []

    for fname in tqdm(fnames):
        # print("Evaluating image: ", fname)
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', 'test', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'annotation', 'test', fname.replace('jpg', 'png')))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        H, W, _ = image.shape
        
        # get the image embedding and flatten it
        img_emb = get_embedding(image, predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        
        # get the mask predicted by the linear classifier
        y_pred = model.predict(img_emb)
        y_pred = y_pred.reshape((64, 64))
        # mask predicted by the linear classifier
        mask_pred_l = cv2.resize(y_pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # use distance transform to find a point inside the mask
        fg_point = get_max_dist_point(mask_pred_l)
        # Define the kernel for dilation
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask_pred_l, kernel, iterations=3)
        mask_pred_l = cv2.dilate(eroded_mask, kernel, iterations=5)
        
        # set the image to sam
        predictor.set_image(image)
        
        # prompt the sam with the point
        input_point = np.array([[fg_point[0], fg_point[1]]])
        input_label = np.array([1])
        masks_pred_sam_prompted1, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )
        
        # prompt the sam with the bounding box
        y_indices, x_indices = np.where(mask_pred_l > 0)
        if np.all(mask_pred_l == 0):
            bbox = np.array([0, 0, H, W])
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask_pred_l.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        masks_pred_sam_prompted2, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,)
            
        # prompt the sam with both the point and bounding box
        masks_pred_sam_prompted3, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bbox[None, :],
            multimask_output=False,)
        
        print(f"mask shape {mask.shape}, mask_pred_sam shape {masks_pred_sam_prompted3[0].shape}")
        dice_l = iou_coef(mask, mask_pred_l)
        dice_p = iou_coef(mask, masks_pred_sam_prompted1[0])
        dice_b = iou_coef(mask, masks_pred_sam_prompted2[0])
        dice_i = iou_coef(mask, masks_pred_sam_prompted3[0])

        TP = np.logical_and(mask, masks_pred_sam_prompted3[0]).sum()
        FP = np.logical_and(np.logical_not(mask), masks_pred_sam_prompted3[0]).sum()
        FN = np.logical_and(mask, np.logical_not(masks_pred_sam_prompted3[0])).sum()

        # Compute IoU
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        iou_scores.append(iou)

        # Compute F1-score
        f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        f1_scores.append(f1)

        dice_linear.append(dice_l)
        dice1.append(dice_p)
        dice2.append(dice_b)
        dice3.append(dice_i)

        # plot the results
        fig, ax = plt.subplots(1, 5, figsize=(15, 10))
        ax[0].set_title('Ground Truth')
        ax[0].imshow(mask)
        ax[1].set_title('Linear + e&d')
        ax[1].plot(fg_point[0], fg_point[1], 'r.')
        ax[1].imshow(mask_pred_l)
        ax[2].set_title('Point')
        ax[2].plot(fg_point[0], fg_point[1], 'r.')
        ax[2].imshow(masks_pred_sam_prompted1[0]) 
        ax[3].set_title('Box')
        show_box(bbox, ax[3])
        ax[3].imshow(masks_pred_sam_prompted2[0])
        ax[4].set_title('Point + Box')
        ax[4].plot(fg_point[0], fg_point[1], 'r.')
        show_box(bbox, ax[4])
        ax[4].imshow(masks_pred_sam_prompted3[0])
        [axi.set_axis_off() for axi in ax.ravel()]
        
        vis_path = os.path.join(args.save_path, 'vis')
        mask_path = os.path.join(args.save_path, 'mask')

        if os.path.exists(vis_path) == False:
            os.mkdir(vis_path)
        if os.path.exists(mask_path) == False:
            os.mkdir(mask_path)
        plt.savefig(os.path.join(vis_path, fname))
        cv2.imwrite(os.path.join(mask_path, fname.replace("jpg", "png")), masks_pred_sam_prompted3[0])
    
    mdice0 = round(sum(dice_linear)/float(len(dice_linear)), 5)
    mdice1 = round(sum(dice1)/float(len(dice1)), 5)
    mdice2 = round(sum(dice2)/float(len(dice2)), 5)
    mdice3 = round(sum(dice3)/float(len(dice3)), 5)
    mf1_score = round(sum(f1_scores)/float(len(f1_scores)), 5)
    miou_score = round(sum(iou_scores)/float(len(iou_scores)), 5)
    
    print('For the first {} images: '.format(num_visualize))
    print('mIou(linear classifier: )', mdice0)
    print('mIou(point prompts): ', mdice1)
    print('mIou(bbox prompts): ', mdice2)
    print('mIou(points and boxes): ', mdice3)
    print('F1(points and boxes): ',  mf1_score)
    print('IoU(points and boxes): ', miou_score)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--data_path', type=str, default='../data', help='path to train data')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/sam_vit_b_01ec64.pth', help='SAM checkpoint')
    parser.add_argument('--visualize', type=bool, default=True, help='visualize the results')
    parser.add_argument('--save_path', type=str, default='../results', help='path to save the results')
    parser.add_argument('--visualize_num', type=int, default=30, help='number of pics to visualize')
    args = parser.parse_args()

    # set random seed
    random.seed(42)
    
    # register the SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    global predictor
    predictor = SamPredictor(sam)
    print('SAM model loaded!', '\n')
    
    models_choice = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Ridge Classifier': RidgeClassifier(),
        # 'SVC': SVC(),
        # 'Random Forest': RandomForestClassifier(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
        # 'Decision Tree': DecisionTreeClassifier(),
        # 'KNN': KNeighborsClassifier()
    }

    for name, model_choie in models_choice.items():
        model = train(args, predictor, model_choie, name)
        test_visualize(args, model, predictor)


if __name__ == '__main__':
    main()