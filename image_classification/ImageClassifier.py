import os


class ImageClassifier:
    def classifyImage(self):
        os.system("python yolov5/detect.py --weights yolov5/runs/train/food_test_various/weights/best.pt --img 640 --conf 0.25 --source yolov5/data/images/rice.jpg")


if __name__ == "__main__":
    imageClassifier = ImageClassifier()
    imageClassifier.classifyImage()