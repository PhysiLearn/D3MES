import dgl
import torch
import torch.nn as nn
import numpy as np
from SE3Transformer.se3_transformer.model.fiber import Fiber
from SE3Transformer.se3_transformer.model.layers.convolution import ConvSE3FuseLevel
from SE3Transformer.se3_transformer.model.layers.attention import AttentionBlockSE3

class XAttention(nn.Module):
    def __init__(
            self,
            num_nodes: int = 75,
            node_fiber_channels: int = 3,
            edge_fiber_channels: int = 3,
            num_heads: int = 1,
            channels_div: int = 2,
            max_degree: int = 4,
            use_layer_norm: bool = False,
            fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
            low_memory: bool = False,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False
    ):
        super(XAttention, self).__init__()
        
        self.num_nodes = num_nodes
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        
        # 定义节点和边的fiber
        self.node_fiber = Fiber({0: node_fiber_channels})
        self.edge_fiber = Fiber({0: edge_fiber_channels})

        # 创建图结构
        edges = [(i, i + 1) for i in range(num_nodes - 1)]
        edges.append((num_nodes - 1, 0))
        src, dst = zip(*edges)
        self.graph = dgl.graph((src, dst)).to('cuda')

        # 定义AttentionBlockSE3实例
        self.attention_block = AttentionBlockSE3(
            fiber_in=self.node_fiber,
            fiber_out=self.node_fiber,
            fiber_edge=self.edge_fiber,
            num_heads=num_heads,
            channels_div=channels_div,
            use_layer_norm=use_layer_norm,
            max_degree=max_degree,
            fuse_level=fuse_level,
            low_memory=low_memory
        )

    def adjust_feature_shape(self, features):
        for key, value in features.items():
            if len(value.shape) == 2:
                features[key] = value.unsqueeze(2).to(value.device)
        return features

    def compute_basis(self, coords, edges):
        num_edges = len(edges)
        basis = torch.zeros((num_edges, 3, 3), device=coords.device)  # 在 GPU 上创建张量
        for i, (src, dst) in enumerate(edges):
            diff = coords[dst] - coords[src]
            norm_diff = diff.norm(p=2) + 1e-6  # 避免除以零
            basis[i] = torch.outer(diff, diff) / norm_diff  # 使用 PyTorch 的 outer 函数
        return basis

    def process_sample(self, node_feat, edge_feat, basis):
        node_features = {'0': node_feat.to('cuda')}
        edge_features = {'0': edge_feat.to('cuda')}
        node_features = self.adjust_feature_shape(node_features)
        edge_features = self.adjust_feature_shape(edge_features)
        output = self.attention_block(node_features, edge_features, self.graph, basis)
        return output['0'].squeeze(2)

    def forward(self, input_data):
        input_data = input_data.to('cuda')  # 确保输入在 GPU 上
        B, C, H, W = input_data.shape
        assert C == 3 and H == self.num_nodes and W == 3, "Input shape should be (B, 3, 27, 3)"
        
        output_data = torch.zeros_like(input_data).to(input_data.device)  # 确保输出张量在 GPU 上

        for i in range(B):
            node_feat = input_data[i, 0].to('cuda')  # 确保 node_feat 在 GPU 上
            edge_feat = input_data[i, 2].to('cuda')  # 确保 edge_feat 在 GPU 上
            coordinates = input_data[i, 0].to('cuda')  # 假设坐标在第一个通道
            
            edges = [(j, j + 1) for j in range(self.num_nodes - 1)] + [(self.num_nodes - 1, 0)]
            basis = {'0': self.compute_basis(coordinates, edges)}  # 计算 basis 保持在 GPU 上
            
            new_node_feat = self.process_sample(node_feat, edge_feat, basis)
            output_data[i, 0] = new_node_feat
            output_data[i, 1] = input_data[i, 1]
            output_data[i, 2] = input_data[i, 2]

        return output_data
