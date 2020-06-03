# #import necessary package
# import socket
# import time
# import sys
# import RPi.GPIO as GPIO
# #define host ip: Rpi's IP
# HOST_IP = "192.168.43.78"
# HOST_PORT = 8888
# print("Starting socket: TCP...")
# #1.create socket object:socket=socket.socket(family,type)
# socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# print("TCP server listen @ %s:%d!" %(HOST_IP, HOST_PORT) )
# host_addr = (HOST_IP, HOST_PORT)
# #2.bind socket to addr:socket.bind(address)
# socket_tcp.bind(host_addr)
# #3.listen connection request:socket.listen(backlog)
# socket_tcp.listen(1)
# #4.waite for client:connection,address=socket.accept()
# socket_con, (client_ip, client_port) = socket_tcp.accept()
# print("Connection accepted from %s." %client_ip)
# socket_con.send(b"Welcome to RPi TCP server!")
# #5.handle
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(11,GPIO.OUT)
# print("Receiving package...")
# while True:
#     try:
#         data=socket_con.recv(512)
#         if len(data)>0:
#             print("Received:%s"%data)
#             if data=='1':
#                 GPIO.output(11,GPIO.HIGH)
#             elif data=='0':
#                 GPIO.output(11,GPIO.LOW)
#             socket_con.send(data)
#             time.sleep(1)
#             continue
#     except Exception:
#             socket_tcp.close()
#             sys.exit(1)
# import socket
from module.wheel_module import WheelModule
import time
wheel = WheelModule(11, 12, 13, 15)
wheel.stop()
from socket import *
ip_port=('192.168.43.78',8080)
back_log=5
buffer_size=1024

tcp_server=socket(AF_INET,SOCK_STREAM)
tcp_server.bind(ip_port)
tcp_server.listen(back_log)

try:
    while True:
        print('服务端开始运行了')
        conn,addr=tcp_server.accept() #服务端阻塞
        print('双向链接是',conn)
        print('客户端地址',addr)

        while True:
            # try:
                data=conn.recv(buffer_size)
                if len(data.decode('utf-8'))>0:
                    print('客户端发来的消息是:',data.decode('utf-8'))
                    if data.decode('utf-8')=='0':
                        wheel.forward()
                        time.sleep(1)
                    elif data.decode('utf-8')=='1':
                        wheel.backward()
                        time.sleep(1)
                    elif data.decode('utf-8')=='2':
                        wheel.left()
                        time.sleep(1)
                    elif data.decode('utf-8')=='3':
                        wheel.right()
                        time.sleep(1)
                    elif data.decode('utf-8')=='4':
                        wheel.stop()
                        time.sleep(1)
                # elif data.decode('utf-8')==0
                conn.send(data.upper())
            # except Exception:
            #     break
except:
    conn.close()
    wheel.quit()
    tcp_server.close()
