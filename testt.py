# cut = [(0, 2), (13, 15), (22, 40), (63, 74), (84, 86), (93, 111), (114, 133)]
# i = 0
# while i < len(cut):
#     (left, right) = cut[i]
#     print(left, right)
#     if right - left < 8:
#         cut.remove((left,right))
#         i -= 1
#     i += 1
# print(cut)


# a = [[1,2,3,4,23],12,123,412,34]
# print(a.flatten())

import numpy as np
# labels = [1,2,3,4]
# pre_result = [1,2,3,3]
# correct = np.equal(labels, pre_result)
# char_correct_rate = np.mean(correct)
# print(char_correct_rate)
# captcha_correct_vector = correct.reshape((-1, 2)).min(axis=1)
# print(captcha_correct_vector)
# captcha_correct_rate = captcha_correct_vector.mean()
# print(captcha_correct_rate)

b = [[0,0],[2,3],[4,0],[0,0]]
# b = np.array(b)
# print(b)
# a = np.where(b[1] - b[0] == 0)
# print(len(a))
# print(a)
# a = a[0]
# print(len(a))
# print(a)
# print(np.mean(a))
b.remove()