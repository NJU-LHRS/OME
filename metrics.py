import numpy as np
from prettytable import PrettyTable

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    def Overall_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        # Acc = np.nanmean(Acc)
        return Acc

    def Recall_Class(self):
        Rec = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Rec = np.nanmean(Rec)
        return Rec

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    def F1_Class(self):
        p, r = self.Pixel_Accuracy_Class(), self.Recall_Class()
        F1 = 2*(p*r) / (p+r)
        return F1

    def Mean_F1(self):
        p, r = self.Pixel_Accuracy_Class(), self.Recall_Class()
        F1 = 2*(p*r) / (p+r)
        mf1 = np.nanmean(F1)
        return mf1

    def _generate_matrix(self, gt_image, pre_img):
        pre_image = pre_img.copy()
        mask = (gt_image >= 0) & (gt_image < self.num_class)        # 用来掩膜掉一部分像元如忽略像元
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # print(gt_image.shape,pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def print_evaluation(val):
    single_res_table = PrettyTable()
    single_res_table.field_names = ['IoU', 'Fscore', 'Precision', 'Recall']  # 表头
    res_list = [np.round(value * 100, 2) for value in
                [val.Intersection_over_Union()[1], val.F1_Class()[1],
                 val.Pixel_Accuracy_Class()[1], val.Recall_Class()[1]]]
    single_res_table.add_row(res_list)
    print('\n' + single_res_table.get_string())

def print_evaluation_dict(dict, disc = ''):
    single_res_table = PrettyTable()
    single_res_table.field_names = [disc]+['IoU', 'Fscore', 'Precision', 'Recall']
    for key, val in dict.items():
        res_list = [np.round(value * 100, 2) for value in
                    [val.Intersection_over_Union()[1], val.F1_Class()[1],
                     val.Pixel_Accuracy_Class()[1], val.Recall_Class()[1]]]
        single_res_table.add_row([key] + res_list)
    print('\n' + single_res_table.get_string())


def evaluate(results, metrics=['mIoU', 'mFscore'], class_names=()):
    from prettytable import PrettyTable
    from collections import OrderedDict
    from mmseg.core.evaluation.metrics import pre_eval_to_metrics
    '''refer to mmsegmentation'''
    eval_results = {}
    ret_metrics = pre_eval_to_metrics(results, metrics)
    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    # each class table
    ret_metrics.pop('aAcc', None)   # Acc==Recall, mAcc==mRecall, aAcc==OA
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print('per class results:')
    print('\n' + class_table_data.get_string())
    print('Summary:')
    print('\n' + summary_table_data.get_string())

    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })

    return eval_results
