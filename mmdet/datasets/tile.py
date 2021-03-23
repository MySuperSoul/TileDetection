import pickle
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TileDataset(CustomDataset):
    CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def load_annotations(self, ann_file: str):
        '''
        return: data_infos [{}] list of dict contains all the annotations
        '''
        assert (ann_file.endswith('pkl'))
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
