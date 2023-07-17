import os.path
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from prettytable import PrettyTable

def calculate_cooccur_matrix(labels_path, num_classes,
                             classes_name = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car']):
    labels = np.load(labels_path,allow_pickle=True).item().values()
    labels = np.array(list(labels))
    mx = np.zeros([num_classes,num_classes])
    for label in tqdm(list(labels)):
        for idx, cls in enumerate(label):
            if cls: mx[idx]+=label
    mx = np.divide(mx, np.diagonal(mx)).T

    single_res_table = PrettyTable()
    single_res_table.field_names = ['cooccur'] + classes_name  # 表头
    for idx, values in enumerate(mx):
        res_list = [np.round(value, 2) for value in values]
        single_res_table.add_row([classes_name[idx]] + res_list)
    print('\n' + single_res_table.get_string())

def generate_uncertainty_from_cam(cam_dir, img_dir, num_classes, out_dir):
    from tool import imutils
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    for cam_path in tqdm(glob(os.path.join(cam_dir,'*.npy'))):
        name = os.path.splitext(os.path.basename(cam_path))[0]
        orig_img = cv2.imread(os.path.join(img_dir, name + '.png'))
        orig_img_size = orig_img.shape[:2]

        cam_dict = np.load(cam_path, allow_pickle=True).item()

        norm_cam = np.zeros([num_classes, orig_img_size[0], orig_img_size[1]])
        for i, cam in cam_dict.items():
            norm_cam[i] = cam

        seg_logit_ori = norm_cam.copy()
        seg_logit = imutils.DenseCRF(orig_img, seg_logit_ori)
        seg_logit = seg_logit.reshape((seg_logit.shape[0], seg_logit.shape[1], seg_logit.shape[2]))
        seg_pred_crf = np.argmax(seg_logit, 0)
        seg_pred_argmax = np.argmax(norm_cam, 0)

        labels = list(set(seg_pred_crf.reshape(seg_pred_crf.shape[0] * seg_pred_crf.shape[1]).tolist()))
        selected = norm_cam[labels]
        uncertainty_maps = []
        for c in range(len(labels)):
            scaled_preds = []
            for n in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
                scaled_pred = selected.copy()
                scaled_pred[c] = scaled_pred[c] ** n
                p = imutils.DenseCRF(orig_img, scaled_pred)
                p = p.reshape((1, p.shape[0], p.shape[1], p.shape[2]))
                p = p.argmax(1)
                scaled_preds.append(p == c)
            uncertainty_map_c = np.concatenate(scaled_preds, 0).var(0)
            uncertainty_maps.append(uncertainty_map_c)

        uncertainty_map = np.stack(uncertainty_maps, 0).max(0)
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / \
                          (uncertainty_map.max() - uncertainty_map.min() + 0.000001)
        weight_map = ((1 - uncertainty_map) * 255).astype('uint8')

        mask_weight = np.stack([seg_pred_argmax, weight_map, seg_pred_crf], 2)
        cv2.imwrite(os.path.join(out_dir, name + '.png'), mask_weight)

        # # entropy = -(norm_cam+1e-8)*np.log(norm_cam+1e-8)
        # entropy = -(norm_cam)*np.log(norm_cam)
        # entropy = np.nanmean(entropy,0)
        # entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
        # orig_img_norm = np.float32(orig_img) / 255
        # hotmap = show_cam_on_image(orig_img_norm, entropy)
        # # out_entropy_heatmap_dir = os.path.join(outdir, 'entropy-heatmap')
        # # if not os.path.exists(out_entropy_heatmap_dir): os.makedirs(out_entropy_heatmap_dir)
        # # cv2.imwrite(os.path.join(out_color_mask_dir, img_name + '.png'), hotmap)
        # out_entropy_dir = os.path.join(outdir, 'entropy')
        # if not os.path.exists(out_entropy_dir): os.makedirs(out_entropy_dir)
        # np.save(os.path.join(out_entropy_dir, img_name + '.npy'), entropy)




def main():

    '''calculate_coocur_matrix'''
    # w = calculate_cooccur_matrix(r'',5,
    #                              classes_name = [])

    '''generate uncertainty from cams'''
    # generate_uncertainty_from_cam(r'',
    #                               r'', 5,
    #                               r'')


    return



if __name__=='__main__':
    main()