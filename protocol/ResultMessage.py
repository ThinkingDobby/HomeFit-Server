import numpy as np

class ResultMessage:
    def __init__(self):
        self.result = None
        self.stateCode = 0

    def getResultMessage(self, messageNumber):
        start = b'['
        end = b']'

        # 분류 및 양 추정 결과 생성
        if messageNumber == 32:
            if self.stateCode == 0:
                msgSize = 5
            else:
                msg5 = bytes(self.result, 'utf-8')
                msgLen = len(msg5)
                msgSize = 5 + msgLen

            # 바이트 변환
            msg2 = np.array(messageNumber).astype('uint8').tobytes()
            msg3 = np.array(msgSize).astype('uint8').tobytes()
            msg4 = np.array(self.stateCode).astype('uint8').tobytes()
            
            if self.stateCode == 0:
                resultMessage = start + msg2 + msg3 + msg4 + end
            else:
                resultMessage = start + msg2 + msg3 + msg4 + msg5 + end
            
            #메시지가 전송 이후 모두 초기화
            self.result = None
            self.stateCode = 0
            return resultMessage
    
    
    def setEstimationResult(self, result, stateCode):
        self.result = result
        self.stateCode = stateCode
