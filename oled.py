from luma.core.interface.serial import i2c, spi
from luma.core.render import canvas
from luma.oled.device import ssd1306
import time

def showOLED(str, sleeptime=3):
    serial = spi(device=0, port=0)

    device = ssd1306(serial)

    with canvas(device) as draw:
      draw.rectangle(device.bounding_box, outline="white", fill="black")
      draw.text((5, 20), str, fill="white")

    #time.sleep(sleeptime)

if __name__ == '__main__':
    showOLED("Hello World!\nFrom OLED TESTING")
