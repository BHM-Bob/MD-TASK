# Optimized from contact_map.py by DeepSeek R1 web-service

import argparse
import os
import sys
from datetime import datetime

import matplotlib
import pandas as pd

if 'MBAPY_PLT_AGG' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mdtraj as md
import networkx as nx


def plot_network(graph, edges_list, output_path, node_size=2900, node_fontsize=9.5,
                 edgewidth_factor=10, edgelabel_fontsize=8):
    """绘制蛋白质接触网络图并保存为PDF"""
    # 确保有边存在
    if not edges_list:
        raise ValueError("没有检测到接触，无法生成网络图")

    # 对边列表按残基编号进行排序
    df_edges = pd.DataFrame(edges_list, columns=["Source", "Target", "Weight"])
    residue_numbers = df_edges["Target"].str.extract(r'(\d+)')[0].astype(int)
    sorted_indices = residue_numbers.argsort()
    sorted_edges = df_edges.iloc[sorted_indices].values

    # 创建图形布局
    plt.figure(figsize=(6.5, 6.5))
    
    # 明确中心节点（所有边的源节点）
    center_node = sorted_edges[0][0]  # 第一个边的源节点
    outer_nodes = [edge[1] for edge in sorted_edges]
    
    # 生成分层布局
    layout = nx.shell_layout(
        graph,
        nlist=[[center_node], outer_nodes],  # 中心节点在第一个shell
        rotate=False  # 保持固定方向
    )

    # 设置可视化参数
    node_colors = ['#B1DF61' if node == center_node else '#B1DF61' 
                   for node in graph.nodes()]
    edge_widths = [graph[u][v]['weight'] * edgewidth_factor 
                   for u, v in graph.edges()]

    # 绘制网络图
    nx.draw_networkx(
        graph,
        pos=layout,
        node_color=node_colors,
        edge_color='lightgrey',
        node_size=node_size,
        width=edge_widths,
        with_labels=True,
        font_family='serif',
        font_size=node_fontsize
    )

    # 添加边权重标签
    edge_labels = {(u, v): f"{graph[u][v]['weight']:.3f}" 
                   for u, v, _ in sorted_edges}
    nx.draw_networkx_edge_labels(
        graph,
        pos=layout,
        edge_labels=edge_labels,
        font_size=edgelabel_fontsize,
        font_family='serif',
        bbox={'facecolor': 'white', 'alpha': 0},
        rotate=False
    )

    # 保存图像
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()


def load_and_preprocess_traj(traj_path, topology, step):
    """加载并预处理分子动力学轨迹
    
    Args:
        traj_path (str): 轨迹文件路径
        topology (str): 拓扑文件路径
        step (int): 轨迹采样步长
    
    Returns:
        md.Trajectory: 处理后的轨迹对象
    """
    try:
        traj = md.load(traj_path, top=topology)[::step]
    except Exception as e:
        sys.exit(f"Error loading trajectory: {str(e)}")

    # 选择CB原子（甘氨酸使用CA）
    atom_indices = [
        atom.index for atom in traj.topology.atoms
        if (atom.name == "CB" and atom.residue.name != "GLY") 
        or (atom.name == "CA" and atom.residue.name == "GLY")
    ]
    return traj.atom_slice(atom_indices)


def calculate_contacts(traj, target_residue, chain, cutoff):
    """计算目标残基的接触频率
    
    Args:
        traj (md.Trajectory): 轨迹对象
        target_residue (str): 目标残基名称（如THR405）
        chain (str): 链标识符
        cutoff (float): 接触距离阈值（纳米）
    
    Returns:
        tuple: (接触字典, 总帧数)
    """
    contacts = {}
    residues = [str(res) for res in traj.top.residues]
    chain_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if target_residue not in residues:
        sys.exit(f"Error: Residue {target_residue} not found")

    for frame in traj:
        # 找到目标残基的原子索引
        target_idx = next(
            i for i, res in enumerate(residues)
            if res == target_residue
        )
        
        # 计算目标原子与其他原子的距离
        distances = md.compute_distances(
            frame, 
            [[target_idx, other_idx] 
             for other_idx in range(len(residues)) 
             if other_idx != target_idx]
        )
        
        # 识别满足距离条件的接触
        for pair_idx, dist in enumerate(distances[0]):
            if dist < cutoff:
                other_idx = pair_idx + (1 if pair_idx >= target_idx else 0)
                other_res = traj.top.residue(other_idx)
                other_chain = chain_chars[other_res.chain.index]
                
                if other_chain == chain:
                    edge = (
                        f"{target_residue}.{chain}",
                        f"{other_res.name}{other_res.resSeq}.{other_chain}"
                    )
                    contacts[edge] = contacts.get(edge, 0) + 1

    return contacts, traj.n_frames


def save_network_data(edges_list, output_csv):
    """将网络数据保存为CSV文件
    
    Args:
        edges_list (list): 边列表数据
        output_csv (str): 输出CSV文件路径
    """
    df = pd.DataFrame(edges_list, columns=["Source", "Target", "Weight"])
    df.to_csv(output_csv, index=False)


def parse_arguments():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有参数的命名空间
    """
    parser = argparse.ArgumentParser(
        description="从分子动力学轨迹生成加权接触网络图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument("trajectory", help="轨迹文件路径")
    parser.add_argument("--topology", required=True,
                       help="拓扑文件路径（当轨迹不包含拓扑信息时必需）")
    parser.add_argument("--residue", required=True,
                       help="中心残基名称（例如：THR405）")

    # 可选参数
    parser.add_argument("--threshold", type=float, default=6.7,
                       help="接触距离阈值（单位：埃，转换为纳米）")
    parser.add_argument("--chain", default="A",
                       help="目标链标识符")
    parser.add_argument("--step", type=int, default=1,
                       help="轨迹帧采样间隔")
    
    # 可视化参数
    parser.add_argument("--nodesize", type=int, default=2900,
                       help="节点大小")
    parser.add_argument("--nodefontsize", type=float, default=9.5,
                       help="节点标签字体大小")
    parser.add_argument("--edgewidthfactor", type=float, default=10.0,
                       help="边宽缩放因子")
    parser.add_argument("--edgelabelfontsize", type=float, default=8.0,
                       help="边标签字体大小")
    parser.add_argument("--discard-graphs", action="store_true",
                       help="不生成网络图")

    return parser.parse_args()


def main(args):
    """主处理流程"""
    # 参数预处理
    args.threshold /= 10  # 将埃转换为纳米
    residue = args.residue.upper()
    prefix = residue.split(".")[0] if "." in residue else residue

    # 1. 加载轨迹数据
    traj = load_and_preprocess_traj(args.trajectory, args.topology, args.step)

    # 2. 计算接触频率
    contacts, n_frames = calculate_contacts(
        traj, residue, args.chain, args.threshold
    )

    # 3. 生成网络图数据
    center_node = f"{residue}.{args.chain}"
    edges_list = [
        [center_node, edge[1], count/n_frames]
        for edge, count in contacts.items()
    ]
    
    # 4. 创建网络图对象
    contact_graph = nx.Graph()
    contact_graph.add_weighted_edges_from(edges_list)

    # 5. 保存输出结果
    output_csv = f"{prefix}_chain{args.chain}_network.csv"
    save_network_data(edges_list, output_csv)

    # 6. 生成可视化图形
    output_png = f"{prefix}_chain{args.chain}_contact_map.png"
    plot_network(
        contact_graph,
        edges_list,
        output_png,
        node_size=args.nodesize,
        node_fontsize=args.nodefontsize,
        edgewidth_factor=args.edgewidthfactor,
        edgelabel_fontsize=args.edgelabelfontsize
    )


if __name__ == "__main__":
    args = parse_arguments()
    start_time = datetime.now()
    main(args)
    print(f"Process completed in {datetime.now() - start_time}")