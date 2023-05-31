import numpy as np

class ResultMessage:
    def __init__(self):
        self.result = None
        self.value = 0

    def getResultMessage(self, messageNumber):
        start = b'['
        end = b']'

        # 분류 및 양 추정 결과 생성
        if messageNumber == 32:
            if self.value == 0:
                msgSize = 6
            else:
                msg6 = bytes(self.result, 'utf-8')
                msgLen = len(msg6)
                msgSize = 6 + msgLen

            # 바이트 변환
            msg2 = np.array(messageNumber).astype('uint8').tobytes()
            msg3 = np.array(msgSize).astype('uint8').tobytes()
            msg4 = np.array(self.value // 2**7).astype('uint8').tobytes()
            msg5 = np.array(self.value % 2**7).astype('uint8').tobytes()
            
            if self.value == 0:
                resultMessage = start + msg2 + msg3 + msg4 + msg5 + end
            else:
                resultMessage = start + msg2 + msg3 + msg4 + msg5 + msg6 + end
            
            # 메시지 전송 이후 초기화
            self.init()
            return resultMessage
        
        elif messageNumber == 33:   # 메시지 전송 후 파일 전송
            msgSize = 6

            # 메시지 번호
            msg2 = np.array(messageNumber).astype('uint8').tobytes()
            
            # 메시지 크기
            msg3 = np.array(msgSize).astype('uint8').tobytes()

            # 전송할 파일 크기 (value)
            msg4 = np.array(self.value // 2**7).astype('uint8').tobytes()
            msg5 = np.array(self.value % 2**7).astype('uint8').tobytes()

            resultMessage = start + msg2 + msg3 + msg4 + msg5 + end

            self.init()
            return resultMessage
    
    
    def setResult(self, result):
        self.result = result

    def setValue(self, value):
        self.value = value

    def init(self):
        self.result = None
        self.value = 0
