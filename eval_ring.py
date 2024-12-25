import numpy as np
import networkx as nx

# 1. 加载数据
data = np.load('noringgen.npz')['data']
print(data.shape) 


third_channel_data = data[:, 2, :, :]


rounded_data = np.round(third_channel_data)


non_zero_data = []
for i in range(rounded_data.shape[0]):
    non_zero_rows = rounded_data[i][~np.all(rounded_data[i] == 0, axis=1)]
    non_zero_data.append(non_zero_rows)

connections = []
for i in range(len(non_zero_data)):
    if non_zero_data[i].shape[1] >= 2:
        connections.append(non_zero_data[i][:, :2])

def has_cycle(connections):
    G = nx.Graph()  
    G.add_edges_from(connections)  
    try:
        cycle = nx.find_cycle(G, orientation='ignore')  
        return True  
    except nx.NetworkXNoCycle:
        return False  

cycles = []
for i in range(0, min(1000, len(connections))):
    cycles.append(has_cycle(connections[i]))

cycles = np.array(cycles)
num_with_cycles = np.sum(cycles) 
print(f"1001-2000个数据中包含闭环的数据数量: {num_with_cycles}")
