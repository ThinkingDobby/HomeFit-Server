import os
import volume_estimation.estimator.demo as demo
class VolumeEstimator:
    def estimateVolume(self, dir, camerainfo):
        return demo.main(dir, camerainfo)
        #os.system("python estimator\demo.py")


if __name__ == "__main__":
    imageClassifier = VolumeEstimator()
    imageClassifier.estimateVolume()