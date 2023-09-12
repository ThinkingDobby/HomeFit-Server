import socket
import threading

from file.FileController import FileController
import os

from protocol.CheckData import checkData
from protocol.ResultMessage import ResultMessage

import image_classification.ImageClassifier as IC
import volume_estimation.VolumeEstimator as VE

IMAGE_DIR_PATH = "image_classification/yolov5/data/images/"

class HomeFitServer:
    def __init__(self, host, port):
        self.servSock = None
        self.clientSock = None
        self.host = host
        self.port = port
        self.bufSize = 1024

    def serverLoop(self):
        self.servSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.servSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.servSock.bind((self.host, self.port))
        self.servSock.listen()
        while True:
            clientSock, addr = self.servSock.accept()
            print("Client Address: ", addr)
            recvThreadHandle = threading.Thread(target=self.recvMsg, args=(clientSock, addr[1]))
            # recvThreadHandle.daemon = True
            recvThreadHandle.start()
            # self.recvMsg(self.clientSock, addr[1])

    def recvMsg(self, clientSocket, sockID):
        userName = None

        while True:
            # 데이터 수신
            data = clientSocket.recv(self.bufSize)
            print(data)

            if not data:
                print("Client Disconnected")
                break

            print("Data Length (" + str(sockID) + "): " + str(len(data)))

            # 데이터 유효성 확인, 메시지 번호 확인
            check = checkData(data)

            # 사용자명 수신
            if check == 1:
                msgSize = data[2]
                userName = data[3:msgSize - 1].decode()

                print("Username Received: " + userName)

            # 파일 수신
            elif check == 2:
                if not userName:
                    print("Username Not Defined")
                    continue

                saveName = "sample.jpeg"

                fileSize = int.from_bytes(data[6:10], byteorder='big', signed=True)
                print("File Size: " + str(fileSize))

                cwd = os.getcwd()
                dirPath = cwd + '/' + IMAGE_DIR_PATH + userName

                imageSaver = FileController(saveName, dirPath, fileSize)
                imageSaver.initImageSaver()

                imageSaver.saveImage(data[11:])

                while True:
                    data = clientSocket.recv(self.bufSize)
                    # print(data)
                    if not data:
                        break

                    fullFlag = imageSaver.saveImage(data)

                    if fullFlag:
                        break

                imageSaver.closeImageSaver()
                del imageSaver

                # 분류 로직
                saveDir = IC.ImageClassifier.classifyImage(self, userName)

                # 양 추정 로직
                VE.VolumeEstimator.estimateVolume(self, saveDir, cameraInfo, userName)

                filePath = str(saveDir) + '/' + "out.json"
                fileSize = os.path.getsize(filePath)

                # 결과 메시지 생성
                resultMessage = ResultMessage()
                resultMessage.setValue(fileSize)
                clientSocket.sendall(resultMessage.getResultMessage(33))

                dataTransferred = 0
                with open(filePath, 'rb') as f:
                    data = f.read(1024)
                    while data:
                        dataTransferred += clientSocket.send(data)
                        data = f.read(1024)

                print("transmission started")
            #카메라 정보 수신
            elif check == 3:
                msgSize = data[2]
                cameraInfo = data[3:msgSize - 1].decode()
                print("cameraInfo Received: " + cameraInfo)

        clientSocket.close()        


if __name__ == "__main__":
    basicTestServer = HomeFitServer("122.38.179.73", 10001)

    print("Server Start")
    basicTestServer.serverLoop()
