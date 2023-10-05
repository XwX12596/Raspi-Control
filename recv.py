import serial


def receiveUSART():
    port = "/dev/ttyUSB0"  # 串口设备路径，根据实际情况修改
    baudrate = 115200  # 波特率，需要与串口设备一致
    timeout = 1  # 超时时间，单位为秒
    input = []
    ser = serial.Serial(port, baudrate, timeout=timeout)
    while True:
        data = ser.read(50)
        if data != b'':
            input.append(int(data[3:7]))
            input.append(int(data[16:20]))
            input.append(int(data[29:33]))
            input.append(int(data[8:12]) - int(data[21:25]))
            input.append(int(data[21:25]) - int(data[34:38]))
            return input


if __name__ == '__main__':
    receiveUSART()
