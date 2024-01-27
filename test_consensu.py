from multiprocessing import Process, Queue, Manager, Barrier
import datetime
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.dates as mdates
import random


MAP_SIZE = 100


def save_msg_vol(file_name, msg_data, comm_iter):
    save_path = os.path.join(os.path.abspath(
                        os.path.join(os.path.abspath(__file__), "..","..")),
                        "data/", 
                        file_name)
    
    with open(save_path, 'a+') as f:
        f.write(f"Communication Iter={comm_iter}\n")
        for agent, msg_nums in msg_data.items():
            write_data = f"agent={agent}\tmsg_nums={msg_nums}\n\n"
            f.write(write_data)
            
def plot_comm_topoloty(workers, comm_alive_dict):
    # 创建一个新的图像
    fig, ax = plt.subplots()

    # 绘制网络图的节点
    for worker in workers:
        circle = Circle(worker.pose, radius=2, edgecolor='black', facecolor='skyblue')  # 创建一个圆形
        ax.add_patch(circle)  # 将圆形添加到图像中
        ax.text(worker.pose[0], worker.pose[1], worker.name, fontsize=6, ha='center', va='center', color='black')  # 绘制标签

    # 绘制网络图的边
    for (worker_name, other_name), alive in comm_alive_dict.items():
        if alive:
            worker_pose = next(worker.pose for worker in workers if worker.name == worker_name)
            other_pose = next(worker.pose for worker in workers if worker.name == other_name)
            ax.plot([worker_pose[0], other_pose[0]], [worker_pose[1], other_pose[1]], color='grey', linewidth=1.5)  # 绘制边

    # 设置坐标轴的范围
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)

    # 移除坐标轴
    ax.axis('off')

    # 显示图像
    plt.pause(3)
    

def plot_agent_value(final_workers):
    
    colors = iter([plt.cm.tab20(i) for i in range(20)])
    fig, ax = plt.subplots()

    for worker in final_workers:
        ax.plot(worker.time_list, worker.comm_iter_list, 
                linewidth = 3,
                color = next(colors),
                label=worker.name
                )


    # 获取最后一个数据的横坐标值
    # 获取最后一个数据的横坐标值
    max_times = [max(worker.time_list) for worker in final_workers]
    last_x = max(max_times)
    # 在最后一个数据上画一条红色虚线
    ax.axvline(x=last_x, color='red', linestyle='--')

    # 在图上添加横坐标值
    ax.text(last_x, 0, str(last_x), color='red', ha='right')


    # 设置标题和坐标轴标签
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .22), loc='lower left',
            ncol=10, mode="expand", borderaxespad=0.,
            frameon=True,
            fancybox=True,
            fontsize=6,
            edgecolor='black')

    # 显示图像
    plt.show()
    


class Worker:
    def __init__(self, i, pose, sleep_time):
        self.id = i
        self.name = f"Agent_{i}"
        self.msg_buffer = []
        self.value = i
        self.pose = pose
        self.time_list = []
        self.comm_iter_list = []
        self.sleep_time = sleep_time
        
    def send_data(self, msgs, queues, name_list, comm_alive_dict):
        # 发送数据到其他进程
        for name in name_list:
            if name != self.name and comm_alive_dict[(self.name, name)]: # 通信状态为 True
                queue = queues[(self.name, name)]
                try:
                    # while not queue.empty():
                        # queue.get_nowait()
                    queue.put(msgs)
                except OSError:
                    print(f"Process {self.name} failed to send message to {name}")
                
    def receive_data(self, queues, name_list, comm_alive_dict):
        self.msg_buffer.clear()
        # 接收数据
        for name in name_list:
            if name != self.name and comm_alive_dict[(name, self.name)]:
                queue = queues[(name, self.name)]
                try:
                    received_msg = []
                    while not queue.empty():
                        received_msg.append(queue.get())
                    # if len(received_msg) >= 2:
                    #     self.msg_buffer.append(received_msg[-1])
                    # elif len(received_msg) == 1:
                    #     self.msg_buffer.extend(received_msg)
                    # else:
                    #     print(f"Process {self.name} received empty message from {name}")
                    self.msg_buffer.extend(received_msg)
                except OSError:
                    print(f"Process {self.name} failed to receive message from {name}")
                    
    def consensus(self): #  find the max value 
        if len(self.msg_buffer) > 0:
            self.value = max(max(self.msg_buffer), self.value)
            time.sleep(self.sleep_time)
    
    def send(self):
        for sender, receiver in comm_alive_dict.keys():
            if sender == self.name and comm_alive_dict[(sender,receiver)]:
                wifi[receiver].append(self.value)
                
    def receive(self):
        self.msg_buffer = []
        self.msg_buffer.extend(wifi[self.name])
        
    def cons(self, start_cal_time): #  find the max value 
        if len(self.msg_buffer) > 0:
            self.value = max(max(self.msg_buffer), self.value)
            self.comm_iter_list.append(self.value)
            time.sleep(self.sleep_time)
            end_cal_time = datetime.datetime.now()
            self.time_list.append((end_cal_time-start_cal_time).total_seconds())

def run_worker(worker, queues, name_list, comm_alive_dict, consensus_list, agent_queue, barrier):
    comm_iter = 0
    # while comm_iter < max_iter:
    start_cal_time = datetime.datetime.now()
    while True:
        # 发送数据
        worker.send_data(worker.value, queues, name_list, comm_alive_dict)

        # 接收数据
        worker.receive_data(queues, name_list, comm_alive_dict)
        
        barrier.wait()
        # 实现consensus
        worker.consensus()
        if worker.value ==0:
            raise("value")
        consensus_list[worker.id - 1] = worker.value
        worker.comm_iter_list.append(worker.value)
        
        
        # 打印接收到的数据
        print(f"Worker {worker.name}, comm_iter: {comm_iter}, consensus value: {consensus_list}\n")
        end_cal_time = datetime.datetime.now()
        worker.time_list.append((end_cal_time - start_cal_time).total_seconds())
        if len(set(consensus_list)) == 1:
           break
        
        comm_iter += 1
    
    # plot_agent_value(time_list, comm_iter_list)
    agent_queue.put(worker)

def multiprocess_queue():
    
    num_worker = 20
    np.random.seed(42)
    # sleep_time = [random.randint(1, 5) for _ in range(num_worker)]
    sleep_time = [1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    print(f"sleep time -------------------- {sleep_time}")
    workers = [Worker(i+1, pose=np.random.rand(2)*MAP_SIZE, sleep_time=sleep_time[i]) for i in range(num_worker)]
    queues = {(worker1.name, worker2.name): Queue() for worker1 in workers for worker2 in workers if worker1 != worker2}
    name_list = [worker.name for worker in workers]
    # comm_alive_dict = Manager().dict({(worker.name, other_worker.name): True if ((worker.id == (other_worker.id-1)) or (worker.id == (other_worker.id + 1))) and (worker.id != other_worker.id) == (other_worker.id+1) else False for worker in workers for other_worker in workers })
    comm_alive_dict = Manager().dict()
    agent_queue = Queue()
    NumAdj = 3
    for worker in workers:
        self_rel_dist = [[other, np.linalg.norm(worker.pose - other.pose)] for other in workers if worker != other]
        soreted_worker_list = sorted(self_rel_dist, key=lambda x: x[1], reverse=False)
        for i in range(len(soreted_worker_list)):
            if i < NumAdj:
                comm_alive_dict[(worker.name, soreted_worker_list[i][0].name)] = True
                comm_alive_dict[(soreted_worker_list[i][0].name, worker.name)] = True
            else:
                comm_alive_dict[(worker.name, soreted_worker_list[i][0].name)] = False  
        
    # plot communication topology
    # plot_comm_topoloty(workers, comm_alive_dict)

    consensus_list = Manager().list()
    consensus_list.extend([0 for _ in range(len(workers))])
    process_time_begin = datetime.datetime.now()
    

    
    processes = []
    final_workers = []
    barrier = Barrier(num_worker)
    for i,worker in enumerate(workers):
        p = Process(target=run_worker, args=(worker, queues, name_list, comm_alive_dict, consensus_list, agent_queue, barrier))
        p.start()
        processes.append(p)
        
    for _ in range(num_worker):
        final_workers.append(agent_queue.get())

    for p in processes:
        p.join()
        
    process_time_end = datetime.datetime.now()
    
    print(f">>>>>>>>>>>>>>>>> process time: {process_time_end - process_time_begin} <<<<<<<<<<<<<<<")

    plot_agent_value(final_workers)


def singleprocess():
    global wifi
    global comm_alive_dict
    num_worker = 20
    NumAdj = 3
    np.random.seed(42)
    # sleep_time = [random.randint(1,5) for _ in range(num_worker)]
    sleep_time = [1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    print(f"sleep time -------------------- {sleep_time}")
    workers = [Worker(i+1, pose=np.random.rand(2)*MAP_SIZE, sleep_time=sleep_time[i]) for i in range(num_worker)]
    wifi = {worker.name:[] for worker in workers}
    comm_alive_dict = {}
    for worker in workers:
        self_rel_dist = [[other, np.linalg.norm(worker.pose - other.pose)] for other in workers if worker != other]
        soreted_worker_list = sorted(self_rel_dist, key=lambda x: x[1], reverse=False)
        for i in range(len(soreted_worker_list)):
            if i < NumAdj:
                comm_alive_dict[(worker.name, soreted_worker_list[i][0].name)] = True
                comm_alive_dict[(soreted_worker_list[i][0].name, worker.name)] = True
            else:
                comm_alive_dict[(worker.name, soreted_worker_list[i][0].name)] = False  
    

    # plot_comm_topoloty(workers, comm_alive_dict)

    comm_iter = 0
    # while comm_iter < max_iter:
    
    start_cal_time = datetime.datetime.now()
    while True:
        print("="*30)
        print(f"comm iteration = {comm_iter}")
        
        for worker in workers:
          worker.send()
        
        for worker in workers:
            worker.receive()
            
        for worker in workers:
            worker.cons(start_cal_time)
        
        agent_comm_cost = [worker.comm_iter_list[-1] for worker in workers]
        print(f"comm_iter: {comm_iter}, consensus value: {agent_comm_cost}\n")
        if len(set(agent_comm_cost)) == 1:
            end_single_time = datetime.datetime.now()
            print(f">>>>>>>>>>>>>>>>> process time: {end_single_time - start_cal_time} <<<<<<<<<<<<<<<")
            break
        
        comm_iter += 1
    
    plot_agent_value(workers)
    

if __name__ == "__main__":
    multiprocess_queue()
    # singleprocess()
        
        
    
    