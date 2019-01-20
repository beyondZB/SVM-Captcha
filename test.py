import time
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
from sklearn import svm
from sklearn.externals import joblib

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
    captcha_image=np.array(captcha_image)
    return captcha_text,captcha_image


def get_next_batch(batch_size=128, image_height=60, image_width=160, max_captcha=4, char_set_len=10):
    st = time.clock()
    batch_x=np.zeros([batch_size,image_height,image_width,3])
    batch_y=np.zeros([batch_size,4])
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_image()
            if image.shape == (60, 160, 3):
                return text, image
            else:
                print("not true")

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        batch_x[i, :] = image
        batch_y[i, :] = list(text)
    ed = time.clock()
    return np.uint8(batch_x), batch_y, ed - st


def show_image(image, grey=False):
    if grey:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def convert_grey(image, flag_show=False):
    # 灰度化
    iimage = Image.fromarray(np.uint8(image))
    # 锐化，很重要，强化边界，使得噪点更容易去除
    iimage = ImageEnhance.Sharpness(iimage).enhance(4)
    # iimage = ImageEnhance.Contrast(iimage).enhance(1)
    iimage = iimage.convert('L')
    image = np.array(iimage)
    if flag_show:
        show_image(image, True)
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


def convert_bin(image, flag_show=False):
    width, height = image.shape
    old_threshold = 0
    new_threshold = 220
    over_threshold_count = 0
    over_threshold_sum = 0
    under_threshold_count = 0
    under_threshold_sum = 0
    # 找到合适的阈值
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

    if flag_show:
        show_image(image, True)
    return np.where(image < new_threshold, 0, 255)


def depoint(image, flag_show=False):
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
    if flag_show:
        show_image(image, True)
    return image


def new_try(image):
    iimage = Image.fromarray(image)
    iimage = ImageEnhance.Sharpness(iimage).enhance(4)
    image = np.array(iimage)
    return image


def vertical_cut(image):
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
    flag_new_cut = False
    cuts = []
    for i,count in enumerate(ver_list):
        # 阈值这里为0
        if flag_new_cut is False and count > 0:
            l = i
            flag_new_cut = True
        if flag_new_cut and count == 0:
            r = i-1
            flag_new_cut = False
            cuts.append((l,r))
    return cuts


def do_vertical_split(image, cut, flag_show=False):
    split_image = []
    for (l,r) in cut:
        split_image.append(image[:, l:r])
        if flag_show:
            show_image(image[:, l:r], True)
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


def image_preprocess(image, flag_show=False):
    if flag_show:
        show_image(image)
    # 转灰度图
    image = convert_grey(image, flag_show)
    # 二值化
    image = convert_bin(image, flag_show)
    # 去燥
    image = depoint(image, flag_show)
    image = depoint(image, flag_show)
    image = depoint(image, flag_show)
    # 找出切割边界
    cut = vertical_cut(image)
    # 去除切割出来的噪点
    remove_point(cut)
    # 判断是否切割成功，若切割出图片不等于4，则预处理失败，直接抛弃这条数据
    error_flag = False
    if len(cut) != 4:
        error_flag = True
        return np.zeros([60,20,4]), error_flag
    # 实施切割
    s_image = do_vertical_split(image, cut, flag_show)
    return s_image, error_flag


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
    correct_rate = np.mean(correct)
    return correct_rate, correct, et - st


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


def train_process(captcha_image_set, captcha_label_set):
    st = time.clock()

    # 图像预处理
    char_img_set, char_label_set, pre_correct_rate, time_data_set_preprocess = \
        data_set_preprocess(captcha_image_set, captcha_label_set)
    print("pre_correct_rate: {:.4f}".format(pre_correct_rate))
    print("time_data_set_preprocess: {:.4f}s".format(time_data_set_preprocess))

    # 提取特征
    feature_vector, time_get_feature_vector = \
        get_feature_vector(char_img_set)
    print("time_get_feature_vector: {:.4f}s".format(time_get_feature_vector))

    # 训练
    svm_model, time_train = train_svm(feature_vector, char_label_set)

    # 在训练集上测试
    char_correct_rate, char_correct_vector, time_test = \
        test_svm(svm_model, feature_vector, char_label_set)
    total_correct_rate = get_total_correct_rate(char_correct_vector, pre_correct_rate)
    print("char_correct_rate:{:.4f}".format(char_correct_rate))
    print("time_test: {:.4f}s".format(time_test))

    # 存储模型
    joblib.dump(svm_model, "./svm_captcha.model")
    print("model saved successfully!")

    ed = time.clock()
    time_total_train = ed - st
    print("time_train: {:.4f}s".format(time_train))
    return svm_model, total_correct_rate, time_total_train


def test_process(captcha_image_set, captcha_label_set, svm_model=None):
    st = time.clock()
    if svm_model == None:
        # 读取模型
        st_ld = time.clock()
        svm_model = joblib.load("./svm_captcha.model")
        ed_ld = time.clock()
        print("time_load_mode: {:.4f}s".format(ed_ld - st_ld))

    # 图像预处理
    char_image_set, char_label_set, pre_correct_rate, time_data_set_preprocess = \
        data_set_preprocess(captcha_image_set, captcha_label_set)
    print("pre_correct_rate:{:.4f}".format(pre_correct_rate))
    print("time_data_set_preprocess: {:.4f}s".format(time_data_set_preprocess))

    # 提取特征
    feature_vector, time_get_feature_vector = \
        get_feature_vector(char_image_set)
    print("time_get_feature_vector: {:.4f}s".format(time_get_feature_vector))

    # 测试
    char_correct_rate, char_correct_vector, time_test = \
        test_svm(svm_model, feature_vector, char_label_set)
    total_correct_rate = get_total_correct_rate(char_correct_vector, pre_correct_rate)
    print("char_correct_rate:{:.4f}".format(char_correct_rate))
    print("time_test: {:.4f}s".format(time_test))
    ed = time.clock()
    time_total_test = ed - st
    return total_correct_rate, time_total_test


def get_total_correct_rate(char_correct_vector, pre_correct_rate):
    captcha_correct_vector = char_correct_vector.reshape((-1, 4)).min(axis=1)
    captcha_correct_rate = captcha_correct_vector.mean()
    total_correct_rate = pre_correct_rate * captcha_correct_rate
    return total_correct_rate


def img_preprocess_demo():
    # 产生一个随机验证码图片
    text, image = gen_captcha_text_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    image_preprocess(image)


def get_train_data_set(need_create_train_set, train_set_size):
    if need_create_train_set:
        # 创建训练集
        captcha_image_set, captcha_label_set, time_get_batch = \
            get_next_batch(batch_size=train_set_size)
        print("time_get_train_batch: {:.4f}s".format(time_get_batch))
        # 存储训练集
        joblib.dump(captcha_image_set, "./train_img.data")
        joblib.dump(captcha_label_set, "./train_label.data")
        print("train data set saved successfully!")
    else:
        # 载入训练集
        captcha_image_set = joblib.load("./train_img.data")
        captcha_label_set = joblib.load("./train_label.data")
    return captcha_image_set, captcha_label_set

def get_test_data_set(need_create_test_set, test_set_size):
    if need_create_test_set:
        # 创建测试集
        captcha_image_set, captcha_label_set, time_get_batch = \
            get_next_batch(batch_size=test_set_size)
        print("time_get_test_batch: {:.4f}s".format(time_get_batch))
        # 存储测试集
        joblib.dump(captcha_image_set, "./test_img.data")
        joblib.dump(captcha_label_set, "./test_label.data")
        print("test data set saved successfully!")
    else:
        # 载入测试集
        captcha_image_set = joblib.load("./test_img.data")
        captcha_label_set = joblib.load("./test_label.data")
    return captcha_image_set, captcha_label_set
        

if __name__ == '__main__':
    # 配置功能=============================================================
    conf_need_demo = True
    conf_need_train = True
    conf_need_create_train_set = True
    conf_need_create_test_set = True
    conf_train_set_size = 2
    conf_test_set_size = 2

    # 展示图片预处理========================================================
    if conf_need_demo:
        img_preprocess_demo()

    # 训练================================================================
    svm_model = None
    if conf_need_train:
        # 获取训练集
        captcha_image_set, captcha_label_set =\
            get_train_data_set(conf_need_create_train_set, conf_train_set_size)

        #开始训练流程
        svm_model, train_correct_rate, train_time = \
            train_process(captcha_image_set, captcha_label_set)
        print("total training time: {:.4f}s.".format(train_time))
        print("totoal train correct rate: {:.4f}.".format(train_correct_rate))
        print("---------------------------------------------------------")

    # 测试================================================================
    # 生成测试集
    captcha_image_set, captcha_label_set= \
        get_test_data_set(conf_need_create_test_set, conf_test_set_size)

    # 开始测试流程
    test_correct_rate, test_time = \
        test_process(captcha_image_set, captcha_label_set, svm_model)
    print("total test time: {:.4f}s.".format(test_time))
    print("totoal test correct rate: {:.4f}.".format(test_correct_rate))
