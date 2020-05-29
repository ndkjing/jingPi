# import socket
# import time
# import sys
# #RPi's IP
# SERVER_IP = "192.168.43.78"
# SERVER_PORT = 8888
#
# print("Starting socket: TCP...")
# server_addr = (SERVER_IP, SERVER_PORT)
# socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# while True:
#     try:
#         print("Connecting to server @ %s:%d..." %(SERVER_IP, SERVER_PORT))
#         socket_tcp.connect(server_addr)
#         break
#     except Exception:
#         print("Can't connect to server,try it latter!")
#         time.sleep(1)
#         continue
# print("Please input 1 or 0 to turn on/off the led!")
# while True:
#     try:
#         data = socket_tcp.recv(512)
#         if len(data)>0:
#             print("Received: %s" % data)
#             command=input(':')
#             print(command)
#             socket_tcp.send(command)
#             time.sleep(1)
#     except Exception:
#         continue
#         socket_tcp.close()
#         socket_tcp=None
#         sys.exit(1)
from socket import *
ip_port=('192.168.43.78',8080)
back_log=5
buffer_size=1024

tcp_client=socket(AF_INET,SOCK_STREAM)
tcp_client.connect(ip_port)

while True:
    msg=input('>>: ').strip()
    if not msg:continue
    tcp_client.send(msg.encode('utf-8'))
    print('客户端已经发送消息')
    data=tcp_client.recv(buffer_size)
    print('收到服务端发来的消息',data.decode('utf-8'))

tcp_client.close()
