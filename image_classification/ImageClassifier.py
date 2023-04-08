import os
import image_classification.yolov5.detect as detect
import HomeFitServer as server


class ImageClassifier:
    def classifyImage(self, username):
        save_dir = detect.run(
            weights = "image_classification/yolov5/runs/train/food_test_various/weights/best.pt",
            source = server.IMAGE_DIR_PATH + username + "/sample.jpeg",
            save_crop= True,
            name= username
        )
        #com = "python image_classification/yolov5/detect.py --weights image_classification/yolov5/runs/train/food_test_various/weights/best.pt --img 640 --conf 0.25 --source image_classification/yolov5/data/images/" + username + "/sample.jpeg --save-crop --name " + username
        #os.system(com)
        return save_dir
        


if __name__ == "__main__":
    imageClassifier = ImageClassifier()
    imageClassifier.classifyImage()