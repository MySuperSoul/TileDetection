import cv2
import imutils
import numpy as np
import os
import concurrent.futures

MAX_MATCHS = 5000
GOOD_RATE = 0.3


def align_imgs(template_img, origin_img):
    '''
    Args:
        template_img: template image
        origin_img: origin image
    Function:
        align the template image to origin image
    '''

    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    origin_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_MATCHS)
    kpsA = orb.detect(template_gray, None)
    kpsB = orb.detect(origin_gray, None)
    # (kpsA, descsA) = orb.detectAndCompute(template_gray, None)
    # (kpsB, descsB) = orb.detectAndCompute(origin_gray, None)

    descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
    kpsA, descsA = descriptor.compute(template_gray, kpsA)
    kpsB, descsB = descriptor.compute(origin_gray, kpsB)

    if len(kpsA) == 0 or len(kpsB) == 0:
        return template_img.copy()

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    # matches = matcher.match(descsA, descsB, None)

    # matches = sorted(matches, key=lambda x: x.distance)

    nn_matches = matcher.knnMatch(descsA, descsB, 2)
    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8  # Nearest neighbor matching ratio

    if len(nn_matches) < 4 or len(nn_matches[0]) == 1:
        return template_img.copy()

    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpsA[m.queryIdx])
            matched2.append(kpsB[m.trainIdx])

    if len(matched1) < 4:
        return template_img.copy()

    ptsA = np.zeros((len(matched1), 2), dtype="float")
    ptsB = np.zeros((len(matched1), 2), dtype="float")

    for i in range(len(matched1)):
        ptsA[i] = matched1[i].pt
        ptsB[i] = matched2[i].pt

    # keep only the top matches
    # keep = int(len(matches) * GOOD_RATE)
    # if keep < 4:
    #     return template_img.copy()
    # matches = matches[:keep]

    # ptsA = np.zeros((len(matches), 2), dtype="float")
    # ptsB = np.zeros((len(matches), 2), dtype="float")

    # # loop over the top matches
    # for (i, m) in enumerate(matches):
    #     # indicate that the two keypoints in the respective images
    #     # map to each other
    #     ptsA[i] = kpsA[m.queryIdx].pt
    #     ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    if H is None:
        return template_img.copy()
    # use the homography matrix to align the images
    (h, w) = origin_img.shape[:2]
    aligned = cv2.warpPerspective(template_img, H, (w, h))

    # copy the black area background from origin image to aligned image
    lower = np.array([0, 0, 0])
    upper = np.array([10, 10, 10])
    shapeMask = cv2.inRange(aligned, lower, upper)
    mask = np.where(shapeMask == 255)

    aligned[mask[0], mask[1], :] = origin_img[mask[0], mask[1], :]
    # shapeMask = np.tile(shapeMask[:, :, None], [1, 1, 3])

    # aligned = aligned + origin_img * shapeMask
    return aligned


def process(img_name):
    template_img_name = '{}_t.jpg'.format(img_name.split('.')[0])
    origin_img = cv2.imread(os.path.join(train_folder, img_name))
    template_img = cv2.imread(os.path.join(template_folder, template_img_name))
    aligned_img = align_imgs(template_img, origin_img)
    cv2.imwrite(os.path.join(saved_folder, template_img_name), aligned_img)
    print('Finish {}'.format(img_name))


if __name__ == '__main__':
    root_path = '/ssd/huangyifei/data_guangdong/tile_round2'
    train_folder = os.path.join(root_path, 'train_imgs')
    template_folder = os.path.join(root_path, 'train_template_imgs')
    saved_folder = '/ssd/huangyifei/data_guangdong/tile_round2/template_aligned_v2'

    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)

    img_names = os.listdir(train_folder)

    data = cv2.imread(
        'data/data_guangdong/tile_round2/train_imgs/272_97_t20201215160809596_CAM1_0.jpg'
    )
    data_t = cv2.imread(
        'data/data_guangdong/tile_round2/train_template_imgs/272_97_t20201215160809596_CAM1_0_t.jpg'
    )

    aligned = align_imgs(data_t, data)

    img = np.vstack([data, aligned])
    cv2.imwrite('test.jpg', img)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    #     for img_name in img_names:
    #         executor.submit(process, img_name)