import socket
import threading

from file.ImageSave import ImageSaver
import os

from protocols.CheckData import checkData


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

                saveName = "sample.png"

                fileSize = int.from_bytes(data[6:10], byteorder='big', signed=True)
                print("File Size: " + str(fileSize))

                cwd = os.getcwd()
                dirPath = cwd + '/images/' + userName

                imageSaver = ImageSaver(saveName, dirPath, fileSize)
                imageSaver.initImageSaver()

                imageSaver.saveImage(data[11:])

                while True:
                    data = clientSocket.recv(self.bufSize)
                    print(data)
                    if not data:
                        break

                    fullFlag = imageSaver.saveImage(data)

                    if fullFlag:
                        break

                imageSaver.closeImageSaver()
                del imageSaver

        clientSocket.close()


if __name__ == "__main__":
    basicTestServer = HomeFitServer("192.168.0.21", 10001)
    print("Server Start")
    basicTestServer.serverLoop()
