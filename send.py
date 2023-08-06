import time
import serial
 
ser = serial.Serial(
    port='/dev/ttyAMA0',
    baudrate=115200,
    parity=serial.PARITY_NONE,#可以不写
    stopbits=serial.STOPBITS_ONE,#可以不写
    bytesize=serial.EIGHTBITS,#可以不写
    timeout=1)
 
while 1:
    ser.write(b'Hello\n')
    time.sleep(1)
