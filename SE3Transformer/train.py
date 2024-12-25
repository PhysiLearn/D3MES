import dgl
import torch
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.layers.convolution import ConvSE3FuseLevel
from se3_transformer.model.layers.attention import AttentionBlockSE3

# 定义节点和边的fiber
node_fiber = Fiber({0: 3})  # 3 channels for degree 0 nodes
edge_fiber = Fiber({0: 27})  # 27 channels for degree 0 edges

# 创建图结构
num_nodes = 27  # 设置为您的节点数量
edges = [(i, i+1) for i in range(num_nodes - 1)]  # 创建边
edges.append((num_nodes - 1, 0))  # 添加连接最后一个节点到第一个节点的边
src, dst = zip(*edges)
graph = dgl.graph((src, dst))

# 定义节点和边的特征
node_features = {
    '0': torch.randn(num_nodes, 3),  # 27 nodes with 3 features each
}
print("graph.num_edges()",graph.num_edges())
edge_features = {
    '0': torch.randn(graph.num_edges(), 27)  # 27 edges with 27 features each
}

# 验证特征形状
for key, value in node_features.items():
    print(f"Node feature {key} shape: {value.shape}")
for key, value in edge_features.items():
    print(f"Edge feature {key} shape: {value.shape}")

# 创建AttentionBlockSE3实例等其他操作

# 定义基（假设已知）
basis = {
    '0': torch.randn(graph.num_edges(), 3, 3),  # Example basis for edge feature 0
}

print("edge_fiber",edge_fiber)
# 创建AttentionBlockSE3实例
attention_block = AttentionBlockSE3(
    fiber_in=node_fiber,
    fiber_out=node_fiber,
    fiber_edge=edge_fiber,
    num_heads=1,
    channels_div=2,
    use_layer_norm=False,
    max_degree=4,
    fuse_level=ConvSE3FuseLevel.FULL,
    low_memory=False
)

# 确保特征符合预期的形状
def adjust_feature_shape(features):
    for key, value in features.items():
        if len(value.shape) == 2:  # 如果是二维张量，扩展为三维张量
            features[key] = value.unsqueeze(2)  # 在最后一个维度增加一个维度
    return features

node_features = adjust_feature_shape(node_features)
edge_features = adjust_feature_shape(edge_features)

# 计算结果
print("input:",node_features)
output = attention_block(node_features, edge_features, graph, basis)
print(output)
