import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from collections import Counter
import math

def get_points(imgpath):
    img = cv2.imread(imgpath)
    pix = img.shape[0:2]
    center = pix[1]//2, pix[0]//2
    size = max(pix)//2, min(pix)//2
    
    # 타원 그리기
    cv2.ellipse(img, center, size, 0, 0, 360, (255, 255, 255))
    pts = cv2.ellipse2Poly(center, size, 0, 0, 360, delta=1)

    for i in range(pix[0]):
        for j in range(pix[1]):
            if(cv2.pointPolygonTest(pts, (j, i), False) < 0):
                img[i][j] =  255 

    # 그레이 스케일로 변환 ---①
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    color = get_color(cv2.cvtColor(imgray, cv2.COLOR_BGR2RGB))
    print("food color : " ,color)

    # 스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전 ---②
    ret, imthres = cv2.threshold(imgray, color, 255, cv2.THRESH_BINARY_INV)
    # 가장 바깥쪽 컨투어에 대해 모든 좌표 반환 ---③
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    food = []
    for i in contour:
        for j in i:
            food.append(j[0])
            
    food = np.array(food).tolist()
    plate_point = np.array(pts).tolist()


    return food, plate_point


def prefix_point(food, plate_point):
    # 좌표상 음수(-) 값들 0으로 조정
    for i,v in enumerate(plate_point):
        if v[0] < 0:
            plate_point[i][0] = 0
        if v[1] < 0:
            plate_point[i][1] = 0

    for i,v in enumerate(food):
        if v[0] < 0:
            food[i][0] = 0
        if v[1] < 0:
            food[i][1] = 0

    return food, plate_point


def create_json(empty_json, result_json, plate, food):
    data = "" 
    with open(empty_json, 'r') as json_file:
        data = json.load(json_file)
    for shape in data['shapes']:
        if(shape['label'] == 'plate'):
            shape['points'] = plate
        else:
            shape['points'] = food
    data['imagePath'] = 'result.jpg'

    with open(result_json, 'w') as json_file:
        json_file.write(json.dumps(data))

def get_color(img):
    data = []
    img = img.reshape(-1, 3)
    for dt in img:
        result = [i if i < 128 else 0 for i in dt ]
        if all(result) > 0:
            data.append(result)

    clt = KMeans(n_clusters=3) # most 3 value 
    clt.fit(data)

    n_pixels = len(clt.labels_)
    counter = Counter(clt.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)

    perc = dict(sorted(perc.items(), key=lambda item : item[1], reverse=True))
    print(perc)
    print(clt.cluster_centers_)
    color = []
    for i, c in enumerate(clt.cluster_centers_):
        color.append(perc[i] * sum(c) / len(c))
    
    return sum(color)
