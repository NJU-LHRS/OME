import numpy as np
import torch
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, imutils, torchutils, visualization
import argparse
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from network import resnet38_SEAM
from MyDataloader import ClsDataset, BinaryClsDataset
import glob

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    # n,c,h,w = x.size()
    # x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    # x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x[x != x_max] = 0
    return x

def worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]

            label = pack[2].cuda(non_blocking=True)

            cam1, cam_rv1 = model(img)
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss = F.multilabel_soft_margin_loss(label1, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    loss = val_loss_meter.pop('loss')
    print('loss: %.4f' % (loss))

    return loss

def train(save_dir):
    model = resnet38_SEAM.Net(num_classes=args.num_classes)

    print(model)

    tblogger = SummaryWriter(save_dir)

    transform_train = transforms.Compose([
        imutils.RandomResizeLong(args.resize[0], args.resize[1]),
        # transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomCrop(args.crop_size, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_val = transforms.Compose([
        transforms.RandomCrop(args.crop_size, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = ClsDataset(args.data_root, args.label_path, transform=transform_train)
    val_dataset = ClsDataset(args.val_data_root, args.val_label_path, transform=transform_val)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        # assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')
    val_min_loss, best_epoch = -1, 0

    timer = pyutils.Timer("Session started: ")

    loss_option = args.loss_option.split('_')
    for ep in range(args.max_epoches):
        print(f'epoch:{ep+1}')
        for iter, pack in enumerate(train_data_loader):

            scale_factor = 0.5
            img1 = pack[1]
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]
            # bg_score = torch.ones((N, 1))
            # label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            # loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label))
            cam1 = F.interpolate(visualization.max_norm(cam1), scale_factor=scale_factor, mode='bilinear',
                                 align_corners=True) * label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1), scale_factor=scale_factor, mode='bilinear',
                                    align_corners=True) * label

            cam2, cam_rv2 = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            # loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label))
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label

            # loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            # loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls1 = F.multilabel_soft_margin_loss(label1, label)
            loss_cls2 = F.multilabel_soft_margin_loss(label2, label)

            if 'cls' in loss_option:
                loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            else: loss_cls = torch.zeros(1).cuda()

            if 'er' in loss_option:
                # cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
                # cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
                # with torch.no_grad():
                #     eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
                # loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))
                loss_er = torch.mean(torch.abs(cam1 - cam2))
            else: loss_er = torch.zeros(1).cuda()

            if 'ecr' in loss_option:
                ns, cs, hs, ws = cam2.size()
                tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
                tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
                loss_ecr1 = torch.mean(
                    torch.topk(tensor_ecr1.view(ns, -1), k=(int)(args.num_classes * hs * ws * 0.2), dim=-1)[0])
                loss_ecr2 = torch.mean(
                    torch.topk(tensor_ecr2.view(ns, -1), k=(int)(args.num_classes * hs * ws * 0.2), dim=-1)[0])
                loss_ecr = loss_ecr1 + loss_ecr2
                loss_ecr = loss_ecr * args.weight_ecr
            else:
                loss_ecr = torch.zeros(1).cuda()

            loss = loss_cls + loss_er + loss_ecr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'tpi: %.4fs' % (timer.get_stage_elapsed() / ((iter + 1) * args.batch_size)),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                img_8 = img1[0].numpy().transpose((1, 2, 0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2, 0, 1))
                h = H // 4
                w = W // 4
                p1 = F.interpolate(cam1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2, (h, w), mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                # MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item()}
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_image('Image', input_img, itr)
                # tblogger.add_image('Mask', MASK, itr)
                tblogger.add_image('CLS1', CLS1, itr)
                tblogger.add_image('CLS2', CLS2, itr)
                tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                tblogger.add_images('CAM1', CAM1, itr)
                tblogger.add_images('CAM2', CAM2, itr)
                tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        if (ep + 1) % args.save_interval == 0: torch.save(model.module.state_dict(), os.path.join(save_dir, f'model_{ep + 1}.pth'))

        new_loss = validate(model, val_data_loader)
        if val_min_loss == -1 or val_min_loss < new_loss:
            val_min_loss = new_loss
            best_epoch = ep + 1
            torch.save(model.module.state_dict(), os.path.join(save_dir, f'model_best.pth'))

        print('')
        timer.reset_stage()

    torch.save(model.module.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    print(f'best epoch with minimal loss ({val_min_loss}): {best_epoch}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", default=r'', type=str)
    # parser.add_argument("--label_path", default=r'', type=str)
    # parser.add_argument("--val_data_root", default=r'', type=str)
    # parser.add_argument("--val_label_path", default=r'', type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--dataset", default='isprs', type=str)
    parser.add_argument("--session_name", default="WSSS", type=str)
    parser.add_argument("--save_interval", default=5, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--resize", default=(128,384), type=int)

    parser.add_argument("--weight_ecr", default=1, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--loss_option", default='cls_ecr', type=str)   # 'cls_ecr'
    parser.add_argument("--weights", default=r'SEAM_model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params',type=str)
    args = parser.parse_args()

    from LoadParameters import load_para_from_dataset
    paras = load_para_from_dataset(args.dataset)
    classdict = paras['class_label']

    runs = sorted(glob.glob(os.path.join('run', args.session_name + '_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    experiment_dir = os.path.join('run', args.session_name + '_{}'.format(str(run_id)))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    pyutils.Logger(os.path.join(experiment_dir, args.session_name + f'_{run_id}.log'))
    pyutils.save_config(experiment_dir, args)
    print(vars(args))

    train(experiment_dir)