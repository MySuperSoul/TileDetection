from util import get_root_path, sliding_crop_canny_imgs, generate_canny_slide_annotations_file, convert_tile_annotations_toinfos
import os

if __name__ == '__main__':
    root_path = get_root_path()
    train_folder = '/work/data/data_guangdong/tile_round2'
    # train_img_dir = os.path.join(train_folder, 'train_imgs')
    # saved_img_dir = os.path.join(train_folder, 'croped_slide_train_win1650')

    # first crop the imgs to slides
    # sliding_crop_canny_imgs(
    #     img_dir=train_img_dir,
    #     save_crop_dir=saved_img_dir,
    #     sliding_win_xsize=1650,
    #     sliding_win_ysize=1650,
    #     overlap=200)

    # second generate the corresponding annotations
    # train_anno_json_name = 'train_canny_slide_win1650_annos.json'
    # bk_anno_json_name = 'bk_canny_slide_win1650_annos.json'
    # all_anno_json_name = 'all_canny_slide_win1650_annos.json'

    # generate_canny_slide_annotations_file(
    #     anno_file=os.path.join(train_folder, 'train_annos.json'),
    #     img_dir=saved_img_dir,
    #     save_anno_folder=train_folder,
    #     train_anno_json_name=train_anno_json_name,
    #     bk_anno_json_name=bk_anno_json_name,
    #     all_anno_json_name=all_anno_json_name)

    # third generate the train anno infos in pickle format
    convert_tile_annotations_toinfos(
        ann_file=os.path.join(train_folder, 'train_annos.json'),
        val_num_rate=0.1,
        infos_save_dir=os.path.join(train_folder, 'infos'),
        train_info_name='train.pkl',
        val_info_name='val.pkl',
        all_info_name='all.pkl')