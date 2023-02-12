import os


class ImageSaver:
    def __init__(self, saveName, dirPath, fileSize):
        self.saveName = saveName
        self.dirPath = dirPath
        self.f = None
        self.nowSize = 0
        self.fileSize = fileSize

    def saveImage(self, data):
        self.nowSize += len(data)
        print("nowSize: " + str(self.nowSize))
        self.f.write(data)
        if self.nowSize >= self.fileSize:
            return True
        else:
            return False

    def initImageSaver(self):
        self.createDirectory()
        self.f = open(self.dirPath + '/' + self.saveName, 'wb')

    def closeImageSaver(self):
        self.f.flush()
        os.fsync(self.f)
        self.f.close()

    def createDirectory(self):
        try:
            if not os.path.exists(self.dirPath):
                os.makedirs(self.dirPath)
        except OSError:
            print("Failed To Create Directory")
