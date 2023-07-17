import cv2
import mmcv
from torch.utils.data import DataLoader
import torchvision
from tool import pyutils, imutils
import argparse
from PIL import Image
import torch.nn.functional as F
from MyDataloader import *
from network import resnet38_SEAM
from tqdm import tqdm
from mmseg.core.evaluation.metrics import intersect_and_union
from metrics import Evaluator, print_evaluation_dict, evaluate
from tool.visualization import generate_result, show_cam_on_image

def crf_with_alpha(orig_img, cam_dict, alpha, withbg):
    v = np.array(list(cam_dict.values()))
    if withbg:
        bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
        bgcam_score = np.concatenate((v, bg_score), axis=0)     # Note: BG is at the end
        crf_score = imutils.DenseCRF(orig_img, bgcam_score)

        n_crf_al = dict()
        n_crf_al[0] = crf_score[0]
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key + 1] = crf_score[i + 1]
    else:
        crf_score = imutils.DenseCRF(orig_img, v)
        n_crf_al = dict()
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key] = crf_score[i]
    return n_crf_al

def crf_with_alpha2(orig_img, cam, alpha):
    bg_score = np.power(1 - cam, alpha)
    bgcam_score = np.concatenate((bg_score[None,:], cam[None,:]), axis=0)
    crf_score = imutils.DenseCRF(orig_img, bgcam_score)
    return crf_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r'', type=str)
    parser.add_argument("--label_dir", default=r'', type=str)
    parser.add_argument("--mask_root", default=r'', type=str)

    parser.add_argument("--weights", default=r"", type=str)
    parser.add_argument("--dataset", default='isprs', type=str)

    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--msf", default=True, type=bool)
    parser.add_argument("--improve_with_cls", default=True, type=bool)
    parser.add_argument("--out_dir_name", default='train', type=str)
    parser.add_argument("--out_cam_npy", default=True, type=bool)
    parser.add_argument("--out_cam_pics", default=True, type=bool)
    parser.add_argument("--out_mask", default=True, type=bool)

    parser.add_argument("--single", default=False, type=list)
    parser.add_argument("--single-print-cam", default=False, type=bool)
    parser.add_argument("--single-print-img", default=False, type=bool)
    parser.add_argument("--thresholds", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], type=list)

    parser.add_argument("--out_crf", default=False, type=bool)
    parser.add_argument("--crf_alpha", default=[4, 24], type=list)

    parser.add_argument("--bg", default=False, type=bool)
    parser.add_argument("--bg_threshold", default=0, type=float)

    args = parser.parse_args()

    # create outdir
    if not args.out_dir_name: args.out_dir_name = os.path.splitext(os.path.basename(args.weights))[0]
    else: args.out_dir_name = os.path.splitext(os.path.basename(args.weights))[0] + '-' +args.out_dir_name
    if args.msf: outdir = os.path.join(os.path.dirname(args.weights),f'{args.out_dir_name}-msf')
    else: outdir = os.path.join(os.path.dirname(args.weights),args.out_dir_name)
    if not os.path.exists(outdir): os.makedirs(outdir)

    # parameters automatically set according to dataset
    from LoadParameters import load_para_from_dataset
    paras = load_para_from_dataset(args.dataset)
    args.num_classes = paras['num_classes']
    args.cls_names = paras['cls_names']
    args.palette = paras['palette']
    args.reduce_zero_label = paras['reduce_zero_label']
    args.single = args.cls_names if args.single else []

    # print args and save log
    pyutils.Logger(os.path.join(outdir, f'{args.out_dir_name}.log'))
    print(args)
    pyutils.save_config(outdir, args)

    # define metrics
    overall_results = []
    if args.out_crf: overall_results_crf = []

    eval_dict = dict()
    for c in args.single:
        eval_dict[c] = dict()
        for th in args.thresholds:
            eval_dict[c][f'thre-{th}'] =Evaluator(2)
        for alpha in args.crf_alpha:
            eval_dict[c][f'crf-{alpha}'] =Evaluator(2)

    # load model and dataset
    model = resnet38_SEAM.Net(args.num_classes)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()

    if args.msf:
        infer_dataset = ClsDatasetMSF(args.data_dir, args.label_dir, scales=[0.5, 1.0, 1.5, 2.0],
                                      inter_transform=torchvision.transforms.Compose([
                                          # np.asarray,
                                          model.normalize,
                                          imutils.HWC_to_CHW]))
    else:
        infer_dataset = ClsDataset(args.data_dir, args.label_dir,
                                      transform=torchvision.transforms.Compose([
                                          # np.asarray,
                                          model.normalize,
                                          imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    for iter, (img_name, input, label) in enumerate(tqdm(infer_data_loader)):
        img_name = img_name[0]
        label = label[0]
        # input = [F.interpolate(i, scale_factor=0.5, mode='bilinear', align_corners=True) for i in input]

        img_path = os.path.join(args.data_dir, img_name + '.png')
        orig_img = cv2.imread(img_path)
        orig_img_size = orig_img.shape[:2]

        if args.msf:
            img_list = input
            def _work(i, img):
                with torch.no_grad():
                    _, cam = model(img.cuda())
                    # cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy()
                    if i % 2 == 1: cam = np.flip(cam, axis=-1)
                    return cam
            thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                                batch_size=12, prefetch_size=0, processes=args.num_workers)
            cam_list = thread_pool.pop_results()
            cam = np.sum(cam_list, axis=0)
        else:
            img = input
            with torch.no_grad():
                _, cam = model(img.cuda())
                cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy()

        if args.improve_with_cls:
            cam = cam * label.clone().view(args.num_classes, 1, 1).numpy()
            
        cam[cam < 0] = 0
        cam_max = np.max(cam, (1, 2), keepdims=True)
        cam_min = np.min(cam, (1, 2), keepdims=True)
        norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(args.num_classes):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam_npy:
            out_cam_npy_dir = os.path.join(outdir, 'cam')
            if not os.path.exists(out_cam_npy_dir): os.makedirs(out_cam_npy_dir)
            np.save(os.path.join(out_cam_npy_dir, img_name + '.npy'), cam_dict)

        if args.bg:
            bg_score = [np.ones_like(norm_cam[0])*args.bg_threshold]
            pred = np.argmax(np.concatenate((norm_cam, bg_score)), 0)
            PALETTE = args.palette+[[0, 0, 0]]
        else:
            pred = np.argmax(norm_cam, 0)
            PALETTE = args.palette

        if args.out_cam_pics:
            out_cam_pics_dir = os.path.join(outdir, 'cam_pred')
            if not os.path.exists(out_cam_pics_dir): os.makedirs(out_cam_pics_dir)

            colormap = generate_result(orig_img, pred, palette=PALETTE)
            cv2.imwrite(os.path.join(out_cam_pics_dir, img_name + '.png'), colormap.astype(np.uint8))

        if args.out_mask:
            out_mask_dir = os.path.join(outdir, 'mask')
            if not os.path.exists(out_mask_dir): os.makedirs(out_mask_dir)
            cv2.imwrite(os.path.join(out_mask_dir, img_name + '.png'), pred.astype(np.uint8))

            color_seg = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for label1, color in enumerate(PALETTE):
                color_seg[pred == label1, :] = color
            out_color_mask_dir = os.path.join(outdir, 'color_mask')
            if not os.path.exists(out_color_mask_dir): os.makedirs(out_color_mask_dir)
            cv2.imwrite(os.path.join(out_color_mask_dir, img_name + '.png'), color_seg[..., ::-1])

        if args.out_crf:
            for t in args.crf_alpha:
                crf_dict = crf_with_alpha(orig_img, cam_dict, t, args.bg)
                out_crf_dir = os.path.join(outdir,'crf', ('%.1f' % t))
                if not os.path.exists(out_crf_dir): os.makedirs(out_crf_dir)
                # np.save(os.path.join(out_crf_dir, img_name + '.npy'), crf)

                crf_mask = np.zeros([args.num_classes+1]+list(orig_img_size)) if args.bg else np.zeros([args.num_classes]+list(orig_img_size))
                for key, map in crf_dict.items():
                    crf_mask[key] = map
                label_crf = np.argmax(crf_mask, 0)
                colormapcrf = generate_result(orig_img, label_crf, palette=PALETTE)
                cv2.imwrite(os.path.join(out_crf_dir, img_name+'.png'), colormapcrf.astype(np.uint8))

        '''accuracy eval'''
        mask_path = os.path.join(args.mask_root, img_name + '.png')
        mask = np.asarray(Image.open(mask_path))
        pre_eval_results = [intersect_and_union(pred, mask, args.num_classes, 255,
                                                reduce_zero_label=args.reduce_zero_label)]
        overall_results.extend(pre_eval_results)

        if args.out_crf:
            crf_eval_results = [intersect_and_union(label_crf, mask, args.num_classes, 255,     # based on the final alpha in list(crf_alpha)
                                                    reduce_zero_label=args.reduce_zero_label)]
            overall_results_crf.extend(crf_eval_results)

        '''single cam'''
        if args.single:
            for i, clsname in enumerate(args.cls_names):
                if not clsname in args.single: continue
                clscam = norm_cam[i]
                if not label[i]: continue
                clsoutdir = os.path.join(outdir, 'single', clsname)
                if args.single_print_cam or args.single_print_img:
                    if not os.path.exists(clsoutdir): os.makedirs(clsoutdir)

                orig_img_norm = np.float32(orig_img) / 255
                colorcam = show_cam_on_image(orig_img_norm, clscam)
                if args.single_print_cam:
                    cv2.imwrite(os.path.join(clsoutdir, img_name + '.png'), colorcam)

                # single-thresholding
                for thre in args.thresholds:
                    thre_res = (clscam>thre).astype(np.uint8)
                    eval_dict[clsname][f'thre-{thre}'].add_batch(thre_res, (mask==(i+int(args.reduce_zero_label))).astype(np.uint8))

                    if args.single_print_img:
                        clsoutdir_thre = os.path.join(clsoutdir, str(thre))
                        if not os.path.exists(clsoutdir_thre): os.makedirs(clsoutdir_thre)
                        cv2.imwrite(os.path.join(clsoutdir_thre, img_name + f'_{thre}.png'), thre_res*255)

                # CRF
                for alpha in args.crf_alpha:
                    crf = crf_with_alpha2(orig_img, clscam, alpha)
                    crf_fg = np.argmax(crf, 0)
                    eval_dict[clsname][f'crf-{alpha}'].add_batch(crf_fg, (mask==(i+int(args.reduce_zero_label))).astype(np.uint8))
                    if args.single_print_img:
                        folder = os.path.join(clsoutdir, ('crf_%.1f' % alpha))
                        if not os.path.exists(folder): os.makedirs(folder)
                        cv2.imwrite(os.path.join(folder, img_name + '.png'), crf_fg.astype(np.uint8)*255)

    # print multi-class accuracy table
    print('Result accuracy:')
    metric = evaluate(overall_results, class_names=args.cls_names)
    metric_dict = dict(metric=metric)
    mmcv.dump(metric_dict, os.path.join(outdir, 'eval.json'), indent=4)

    if args.out_crf:
        print('Result crf accuracy:')
        metric_crf = evaluate(overall_results_crf, class_names=args.cls_names)
        metric_crf_dict = dict(metric=metric_crf)
        mmcv.dump(metric_crf_dict, os.path.join(outdir, 'eval_crf.json'), indent=4)

    # print single-class accuracy table
    for i, clsname in enumerate(args.cls_names):
        if not clsname in args.single: continue
        print_evaluation_dict(eval_dict[clsname], disc=clsname)
