import argparse
import torch
import cv2
import os
import torch.nn.parallel
from . import modules
from . import net
from . import resnet
from . import densenet
from . import senet

import numpy as np
from . import loaddata_demo as loaddata
import pdb
import argparse
from volume_estimation.estimator.volume import get_volume
from volume_estimation.estimator.mask import get_mask
from volume_estimation.estimator.makejson import prefix_point
from volume_estimation.estimator.makejson import get_points
from volume_estimation.estimator.makejson import create_json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.image
import matplotlib.pyplot as plt
import json

from threading import Thread

parser = argparse.ArgumentParser(description='KD-network')
#parser.add_argument('--img', metavar='DIR',default="volume_estimation\estimator\input\result.jpg",
#                    help='img to input')
parser.add_argument('--json', metavar='DIR',default="volume_estimation/estimator/empty.json",
                    help='json file to input')
parser.add_argument('--resultjson', metavar='DIR',default="volume_estimation/estimator/input/result.json",
                    help='json file to input')
parser.add_argument('--output', metavar='DIR',default="volume_estimation/estimator/output",
                    help='dir to output')

args=parser.parse_args()

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        print(1)
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        print(2)
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

def calculate_nutrition(path, model, nutrition_json, convert_json):
        img = cv2.imread(path)
        nyu2_loader = loaddata.readNyu2(path)
        vol = test(nyu2_loader, model, img.shape[1], img.shape[0], path)
        food_name = str(os.path.dirname(path)).split("\\")[-1]
        print(food_name)
        key = list(vol.keys())[0]

        volume = vol.pop(key)
        nutrition_info = nutrition_json[food_name]

        weight = nutrition_info["conversion_value"] * volume
        calorie = nutrition_info["calorie"] * weight
        fat = nutrition_info["fat"] * weight
        carbohydrate = nutrition_info["carbohydrate"] * weight
        protein = nutrition_info["protein"] * weight

        convert_json[food_name] = {
            "volume(cm^3)" : volume,
            "weight(g)" : weight,
            "calorie(kcal)" : calorie,
            "fat(g)" : fat,
            "carbohydrate(g)" : carbohydrate,
            "protein(g)" : protein
        }
   
def main(dir):
    output_image = str(dir) + "/sample.png"
    crop_imagesDIR = str(dir) + "\crops"
    
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./pretrained_model/model_senet', map_location = "cpu") )
    model.eval()
    
    
    #crop images to array
    crop_images_arr = []
    for root, dirs, files in os.walk(crop_imagesDIR):
        if not dirs:
            crop_images_arr.append(root + "\\" + files[0])

    with open("./volume_estimation/nutrition.json", "r") as file:
        nutrition_json = json.load(file)    
    
    convert_json = {}

    threads = []
    for path in crop_images_arr:
        thread = Thread(target=calculate_nutrition, args=(path, model, nutrition_json, convert_json))
        thread.start()
        threads.append(thread)

    # 모든 스레드가 완료까지 대기
    for thread in threads:
        thread.join()

    #out.json파일에 결과 저장
    with open(str(dir)+"\\out.json", 'a') as out_file:
        json.dump(convert_json, out_file, ensure_ascii=False, indent="\t")
    


def test(nyu2_loader, model, width, height, path):
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True)
        out = model(image)
        out = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        max_pix = out.max() 
        min_pix = out.min()
        out = (out-min_pix)/(max_pix-min_pix)*255
        out = cv2.resize(out,(width,height),interpolation=cv2.INTER_CUBIC)
        
        out_path = str(os.path.dirname(path))
        cv2.imwrite(os.path.join(out_path, "out_grey.png"), out)
        out_grey = cv2.imread(os.path.join(out_path, "out_grey.png"),0)
        out_color = cv2.applyColorMap(out_grey, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_path, "out_color.png"),out_color)
        
        # json create section
        food, plate_point = get_points(os.path.join(out_path, "out_grey.png"))
        food, plate_point = prefix_point(food, plate_point, )
        create_json(args.json, args.resultjson, plate_point, food)
        
        # volume estimation
        vol = get_volume(out_grey, args.resultjson)
        print("\nVolume result :",end="")
        print(vol)
        print("unit: cm^3\n")
        get_mask(out_grey, args.resultjson, out_path)
        return vol
        
if __name__ == '__main__':
    main()