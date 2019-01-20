import time
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
import DropFall
from sklearn import svm
import mnist_unpack

number=['0','1','2','3','4','5','6','7','8','9']
#alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def random_captcha_text(char_set=number,captcha_size=4):
    captcha_text=[]
    for i in range(captcha_size):
        c=random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_image():
    image=ImageCaptcha(font_sizes=(56, 57))
    captcha_text=random_captcha_text()
    captcha_text=''.join(captcha_text)
    captcha_image=image.generate_image(captcha_text)
    # print(captcha_image)
    # captcha_image = captcha_image.convert('L')
    captcha_image=np.array(captcha_image)
    return captcha_text,captcha_image


# def text2vec(text, max_captcha=4, char_set_len=10):
#     text_len = len(text)
#     if text_len > max_captcha:
#         raise ValueError('验证码最长4个字符')
#
#     vector = np.zeros(max_captcha * char_set_len)
#
#     def char2pos(c):
#         if c == '_':
#             k = 62
#             return k
#         k = ord(c) - 48
#         if k > 9:
#             k = ord(c) - 55
#             if k > 35:
#                 k = ord(c) - 61
#                 if k > 61:
#                     raise ValueError('No Map')
#         return k
#
#     for i, c in enumerate(text):
#         idx = i * char_set_len + char2pos(c)
#         vector[idx] = 1
#     return vector


def get_next_batch(batch_size=128, image_height=60, image_width=160, max_captcha=4, char_set_len=10):
    st = time.clock()
    batch_x=np.zeros([batch_size,image_height,image_width,3])
    batch_y=np.zeros([batch_size,4])
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_image()
            # image = np.uint8(image)
            if image.shape == (60, 160, 3):
                return text, image
            else:
                print("not true")

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        # image = convert_grey(image)
        batch_x[i, :] = image
        # batch_y[i, :] = text2vec(text)
        batch_y[i, :] = list(text)
    ed = time.clock()
    return batch_x, batch_y, ed - st


def show_image(image, grey=False):
    # print(image.shape)
    # print(image)
    if grey:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def convert_grey(image):
    iimage = Image.fromarray(np.uint8(image))
    # iimage = iimage.filter(ImageFilter.MedianFilter())
    iimage = ImageEnhance.Sharpness(iimage).enhance(4)
    # iimage = ImageEnhance.Contrast(iimage).enhance(1)
    iimage = iimage.convert('L')
    image = np.array(iimage)
    return image


def pixel_chek(image, pi, pj, chek):
    width, height = image.shape
    if pi < 0 or pi >= width:
        return False
    if pj < 0 or pj >= height:
        return False
    return chek(image[pi, pj])


def pixel_around_check(image, pi, pj, check):
    count = 0
    for i_shift in [-1, 0, 1]:
        for j_shift in [-1, 0, 1]:
            if i_shift == 0 and j_shift == 0:
                continue
            if pixel_chek(image, pi + i_shift, pj + j_shift, check):
                count += 1
    return count


def get_pixel_around(image, pi, pj):
    window = []
    for i_shift in [-1, 0, 1]:
        for j_shift in [-1, 0, 1]:
            if pixel_chek(image, pi + i_shift, pj + j_shift, lambda pix: True):
                window.append(image[pi + i_shift, pj + j_shift])
    return window


def convert_bin(image):
    width, height = image.shape
    old_threshold = 0
    new_threshold = 220
    over_threshold_count = 0
    over_threshold_sum = 0
    under_threshold_count = 0
    under_threshold_sum = 0

    while old_threshold != new_threshold:
        old_threshold = new_threshold
        for i in range(width):
            for j in range(height):
                if image[i][j] > old_threshold:
                    over_threshold_count += 1
                    over_threshold_sum += image[i][j]
                else:
                    under_threshold_count += 1
                    under_threshold_sum += image[i][j]

        avg_ot = over_threshold_sum / over_threshold_count
        avg_ut = under_threshold_sum/ under_threshold_count
        new_threshold = (avg_ot + avg_ut) // 2
        # print(old_threshold, under_threshold_count, over_threshold_count, new_threshold)

    return np.where(image < new_threshold, 0, 255)


def depoint(image):
    """传入二值化后的图片进行降噪"""
    h, w = image.shape
    new_image = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            count = pixel_around_check(image, j, i, lambda pix: pix > 240)
            if count > 4:
                image[j,i] = 255
            if (j in (0, w -1) or i in (0, w - 1)) and count > 2:
                image[j,i] = 255

    return image

def new_try(image):
    iimage = Image.fromarray(image)
    iimage = ImageEnhance.Sharpness(iimage).enhance(4)
    image = np.array(iimage)
    return image


def vertical(image):
    """传入二值化后的图片进行垂直投影"""
    h,w = image.shape
    # print(h, w)
    ver_list = []
    # 开始投影
    for x in range(w):
        black = 0
        for y in range(h):
            if image[y,x] == 0:
                black += 1
        ver_list.append(black)
    # print(ver_list)
    # 判断边界
    l,r = 0,0
    flag = False
    cuts = []
    for i,count in enumerate(ver_list):
        # 阈值这里为0
        if flag is False and count > 0:
            l = i
            flag = True
        if flag and count == 0:
            r = i-1
            flag = False
            cuts.append((l,r))
    return cuts


def do_vertical_split(image, cut):
    split_image = []
    for (l,r) in cut:
        # if (r - l) >= 27:
        #     # split_image.append(image[:, l:(l+r)//2])
        #     # split_image.append(image[:, (l+r)//2 + 1:r])
        #     split_image.extend(DropFall.drop_fall(image[:, l:r]))
        #     # [first_img, second_img] = DropFall.drop_fall(image[:, l:r])
        #     # first_img_h, first_img_w = first_img.shape
        #     # second_img_h, second_img_w = second_img.shape
        #     # if first_img_w >= 30:
        #     #     print("cut_first")
        #     #     split_image.extend(DropFall.drop_fall(first_img))
        #     # if second_img_w >=30:
        #     #     print("cut_second")
        #     #     split_image.extend(DropFall.drop_fall(second_img))
        # else:
        #     split_image.append(image[:, l:r])
        split_image.append(image[:, l:r])
    return split_image


def remove_point(cut):
    i = 0
    while i < len(cut):
        (left, right) = cut[i]
        if right - left < 8:
            cut.remove((left, right))
            i -= 1
        i += 1
    return cut


def image_preprocess(image):
    # show_image(image)

    image = convert_grey(image)
    # show_image(image, True)

    # image = new_try(image)
    # show_image(image, True)

    image = convert_bin(image)
    # show_image(image, True)

    image = depoint(image)
    # show_image(image, True)

    image = depoint(image)
    # show_image(image, True)

    image = depoint(image)
    # show_image(image, True)

    cut = vertical(image)
    # print(cut)

    remove_point(cut)
    # print(cut)
    error_flag = False
    if len(cut) != 4:
        error_flag = True
        return np.zeros([60,20,4]), error_flag

    s_image = do_vertical_split(image, cut)
    # print("=========")
    # for im in s_image:
    #     print(im.shape)
    #     show_image(im, True)

    # # ii = Image.fromarray(np.uint8(image))
    # contours, hierarchy = cv2.findContours(cv2.Umat(image), cv2.RETR_TREE, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # show_image(ii, True)
    return s_image, error_flag


# 读取训练数据
def read_all_data(path, kind='train'):
    if kind == 'train':
        return mnist_unpack.mnist_unpack(path)
    else:
        return mnist_unpack.mnist_unpack(path, 't10k')


def train_svm(images, labels, decision='ovr'):
    st = time.clock()
    clf = svm.SVC(decision_function_shape=decision, gamma='auto')
    clf.fit(np.int8(images), np.int8(labels))
    et = time.clock()
    return clf, et - st


def test_svm(clf, images, labels):
    st = time.clock()
    pre_result = clf.predict(np.uint8(images))
    et = time.clock()
    correct = np.equal(labels, pre_result)
    print("test_svm mean corret len: ", len(correct))
    correct_rate = np.mean(correct)
    return correct_rate, correct, et - st


# 对10个数字进行分类测试
def main():
    data_path = \
        'E:\Workspace\Python\machine-learning\machine-learning\svm\digit_handwritten_recognition\MNIST_28x28'
    # train=========================================================
    train_images, train_labels = read_all_data(data_path, 'train')
    r_train_images = np.round(train_images / 255)  # 对灰度值进行归一化
    print("training...")
    svm_model, train_time = train_svm(r_train_images, train_labels)
    print("training time: {:.4f}s.".format(train_time))
    print("---------------------------------------------------------")
    # test==========================================================
    test_images, test_labels = read_all_data(data_path, 'test')
    r_test_images = np.round(test_images / 255)  # 对灰度值进行归一化
    print("testing...")
    correct_rate, test_time = test_svm(svm_model, r_test_images, test_labels)
    print("test time: {:.4f}s.".format(test_time))
    print("correct_rate: {}.".format(correct_rate))


def data_set_preprocess(captcha_image_set, train_captcha_label_set):
    st = time.clock()
    character_image_set = []
    character_label_set = []
    pre_error_count = 0
    for i, captcha_image in enumerate(captcha_image_set):
        char_image_set, error_flag = image_preprocess(captcha_image)
        if error_flag == False:
            character_image_set.extend(char_image_set)
            character_label_set.extend(train_captcha_label_set[i])
        else:
            pre_error_count += 1
    # print("!!!!!!!!!!!!!!!!!pre_error_count: ", pre_error_count)
    pre_correct_rate = 1 - pre_error_count / len(captcha_image_set)
    ed = time.clock()
    return character_image_set, np.array(character_label_set), pre_correct_rate, ed - st


def get_feature_vector(char_img_set, uniform_width=20):
    st = time.clock()
    vector_set = []
    for char_img in char_img_set:
        height, width = char_img.shape
        vector = []
        for row in char_img:
            black_pos = np.where(row == 0)[0]
            count_blk_pos = len(black_pos)
            uniform_count_blk_pos = count_blk_pos * uniform_width // width
            avg_blk_pos = 0
            uniform_avg_blk_pos = 0
            # print("get_feature_vector black_pos mean lean: ", count_blk_pos)
            if count_blk_pos!= 0:
                avg_blk_pos = np.mean(black_pos)
                uniform_avg_blk_pos = avg_blk_pos * uniform_width // width
            # print(uniform_count_blk_pos)
            # print(uniform_avg_blk_pos)
            vector.append(uniform_count_blk_pos)
            vector.append(uniform_avg_blk_pos)
        vector_set.append(vector)
    ed = time.clock()
    return np.array(vector_set), ed - st


def train(train_set_size = 15000):
    st = time.clock()
    # 训练
    captcha_image_set, captcha_label_set, time_get_batch = \
        get_next_batch(batch_size=train_set_size)
    captcha_image_set = np.uint8(captcha_image_set)
    # print(captcha_image_set[0])
    # show_image(np.uint8(captcha_image_set[0]))
    # print(captcha_image_set)
    # print(captcha_label_set)
    # captcha_label_set = captcha_label_set.flatten()
    # print(captcha_label_set)
    # 图像预处理
    char_img_set, char_label_set, pre_correct_rate, time_data_set_preprocess = \
        data_set_preprocess(captcha_image_set, captcha_label_set)
    # 提取特征
    feature_vector, time_get_feature_vector = \
        get_feature_vector(char_img_set)
    # 训练
    print(feature_vector.shape, char_label_set.shape)
    print(char_label_set)
    svm_model, time_train = train_svm(feature_vector, char_label_set)
    ed = time.clock()
    time_total_train = ed - st
    print("pre_correct_rate: {:.4f}".format(pre_correct_rate))
    print("time_get_batch: {:.4f}s".format(time_get_batch))
    print("time_data_set_preprocess: {:.4f}s".format(time_data_set_preprocess))
    print("time_get_feature_vector: {:.4f}s".format(time_get_feature_vector))
    print("time_train: {:.4f}s".format(time_train))
    return svm_model, pre_correct_rate, time_total_train


def test(svm_model, test_set_size=2500):
    st = time.clock()
    # 测试
    captcha_image_set, captcha_label_set, time_get_batch = \
        get_next_batch(batch_size=test_set_size)
    captcha_image_set = np.uint8(captcha_image_set)
    # 图像预处理
    char_image_set, char_label_set, pre_correct_rate, time_data_set_preprocess = \
        data_set_preprocess(captcha_image_set, captcha_label_set)
    # 提取特征
    feature_vector, time_get_feature_vector = \
        get_feature_vector(char_image_set)
    # 测试
    print(feature_vector.shape, char_label_set.shape)
    print(char_label_set)
    char_correct_rate, char_correct_vector, time_test = \
        test_svm(svm_model, feature_vector, char_label_set)
    captcha_correct_vector = char_correct_vector.reshape((-1, 4)).min(axis=1)
    captcha_correct_rate = captcha_correct_vector.mean()
    total_correct_rate = pre_correct_rate * captcha_correct_rate
    ed = time.clock()
    time_total_test = ed - st
    print("test mean captcha_correct_vector len: ", len(captcha_correct_vector))
    print("pre_correct_rate:{:.4f}".format(pre_correct_rate))
    print("time_get_batch: {:.4f}s".format(time_get_batch))
    print("time_data_set_preprocess: {:.4f}s".format(time_data_set_preprocess))
    print("time_get_feature_vector: {:.4f}s".format(time_get_feature_vector))
    print("time_test: {:.4f}s".format(time_test))
    return total_correct_rate, time_total_test


def img_preprocess_demo():
    text, image = gen_captcha_text_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    image_preprocess(image)


if __name__ == '__main__':
    # 展示图片预处理
    img_preprocess_demo()

    # 训练
    svm_model, pre_correct_rate, train_time = train()
    print("total training time: {:.4f}s.".format(train_time))
    print("---------------------------------------------------------")

    # 测试
    correct_rate, test_time = test(svm_model)
    print("total test time: {:.4f}s.".format(test_time))
    print("correct_rate: {}.".format(correct_rate))
