


def load_para_from_dataset(datasetname):
    para_dict = {}
    if datasetname == 'isprs':
        para_dict['num_classes'] = 5
        para_dict['cls_names'] = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car']
        para_dict['palette'] = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]]
        para_dict['reduce_zero_label'] = True       # index 0 in ground-truth is clutter
        para_dict['class_label'] = {'imper': 0, 'building': 1, 'lowveg': 2, 'tree': 3, 'car': 4}
    if datasetname == 'deepglobe':
        para_dict['num_classes'] = 7
        para_dict['cls_names'] = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown']
        para_dict['palette'] = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255],
                              [0, 0, 0]]
        para_dict['reduce_zero_label'] = False
        para_dict['class_label'] = {'urban': 0, 'agriculture': 1, 'rangeland': 2, 'forest': 3, 'water': 4, 'barren': 5, 'unknown': 6}
    if datasetname == 'landcoverai':
        para_dict['num_classes'] = 4
        para_dict['cls_names'] = ['building', 'tree', 'water', 'road']
        para_dict['palette'] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
        para_dict['reduce_zero_label'] = True
    return para_dict