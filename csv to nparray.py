

import numpy as np
import csv
from sklearn.model_selection import train_test_split

data=[]

#오름차순 정렬된 csv 파일 입력
file = open('H:/1. Research/3. PoDoc_Kim/1. CoCr_AI design/2. Python Code/n_comp_n_hatch_CODE_NPY/3_8.csv', 'r', encoding = 'utf-8')


reader = csv.reader(file)

for line in reader :
    data.append(line[1:])
file.close()

xy=np.array(data[2:])
xy=xy.astype(float)


# 라벨링 구간

numb = 1500 
# n_5 --> 300
# n_8 --> 1500
# n_15 --> 2500


xy_vsplit = np.vsplit(xy, [numb]) # 상위 하위 나누는 값 입력

label1 = xy_vsplit[0] # label1 = 상위 numb개의 값
label2 = xy_vsplit[1] # label2 = 하위 모든 값

XY_spliter1 = np.hsplit(label1, [3]) # 7번째 열(Response 값)을 기준으로 어레이 분할
XY_spliter2 = np.hsplit(label2, [3])



X_label1 = XY_spliter1[0]
Y_label1 = XY_spliter1[1]

X_label2 = XY_spliter2[0]
Y_label2 = XY_spliter2[1]

Y_label1[::] = 0 # 우수한 label을 0으로 지정
Y_label2[::] = 1 # 열등한 label을 1로 지정

#8번째 열은 7번째 열과 반대의 값

zeros_1 = np.zeros((numb,1))
zeros_1[::] = 1
zeros_2 = np.zeros((len(data)-numb-2,1))

Y_label1 = np.concatenate([Y_label1,zeros_1], axis=-1)
Y_label2 = np.concatenate([Y_label2,zeros_2], axis=-1)


X = np.concatenate([X_label1,X_label2], axis=0) # 분할한 array를 병합하여 트레인 데이터 생성
Y = np.concatenate([Y_label1,Y_label2], axis=0) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy_end = (X_train, X_test, Y_train, Y_test)




np.save("H:/1. Research/3. PoDoc_Kim/1. CoCr_AI design/2. Python Code/n_comp_n_hatch_CODE_NPY/3_8", xy_end)

print("ok", len(Y))
