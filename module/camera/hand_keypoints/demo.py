# python主程序也属于一个线程
import threading #线程模块
import time

def Hi(num):       
    print("hello %d\n"%num)
    time.sleep(3)

if __name__ == '__main__':

    t1=threading.Thread(target=Hi,args=(10,))#创建了一个线程对象t1
    t1.start()          # 运行线程

    t2 = threading.Thread(target=Hi, args=(9,))  # 创建了一个线程对象t1
    t2.start()       # 运行线程

    print("ending..........") #   主线程
#hello 10   
#hello 9  同时输出  hello9 hello10，ending
#ending..........  过3秒后程序结束