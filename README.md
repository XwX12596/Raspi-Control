# Raspi-Control
STM32 => RaspberryPi(NN + OLED)

## Deploy
1. RaspberryPi
    - /boot/config.txt
        - enable_uart=1 
        - dtoverlay=pi3-miniuart-bt
    - minicom
        - sudo minicom -D /dev/ttyUSB0 -b 115200
    - network
        - sudo systemctl start(enable) NetworkManager
        - nmtui
    - raspi-config
        - autologin
        - ensure serial
        - enable spi
    - permissions
        - use pi or add `your user` into `kmem`, `gpio`, `spi`, `tty` group
        - sudo chmod 666 /dev/ttyAMA0
2. Python
    - pip
        - torch
        - torchvision
        - collection
        - sudo -H pip install --upgrade luma.oled
    - apt
        - python3-serial
        - libfreetype6-dev libjpeg-dev build-essential
3. Systemd
```
[Unit]
Description=raspi-control
After=multi-user.target

[Service]
User=xmh
Type=simple
ExecStart=/usr/bin/python /home/xwx/raspi-control/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```
