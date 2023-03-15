def checkData(data):
    if data[0] == ord('['):
        print("Correct Message Start")
    else:
        print("Wrong Message Start")
        return 0
    try:
        if data[data[2] - 1] == ord(']'):
            print("Correct Message End")
        else:
            print("Wrong Message End")
            return 0
    except IndexError as e:
        print("Wrong Message Length")
        return 0

    if data[1] == 1:
        return 1
    elif data[1] == 2:
        return 2
    else:
        print("Message Number Not Defined")
        return 0

