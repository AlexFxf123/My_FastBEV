import numpy as np
import struct
import matplotlib.pyplot as plt
import os

def parse_pcd_binary_data(data, point_size=43):
    """
    解析PCD二进制数据，根据实际数据长度计算点数
    
    参数:
        data: 二进制数据
        point_size: 每个点的字节数
    
    返回:
        包含所有点云数据的字典列表
    """
    # 根据实际数据长度计算点数
    actual_bytes = len(data)
    num_points = actual_bytes // point_size
    
    print(f"实际数据字节数: {actual_bytes}")
    print(f"每个点字节数: {point_size}")
    print(f"根据数据计算的点数: {num_points}")
    
    if num_points == 0:
        print("警告: 没有足够的数据解析点云")
        return []
    
    # 解析二进制数据
    raw = np.frombuffer(data[:num_points*point_size], dtype=np.uint8).reshape(-1, point_size)
    print(f"解析形状: {raw.shape}")
    
    # 提取所有点云数据
    points = []
    
    for i in range(raw.shape[0]):
        # 使用struct解析二进制数据
        point_bytes = bytes(raw[i])
        
        # 解析单个点
        point = {
            'index': i + 1,
            'x': struct.unpack('f', point_bytes[0:4])[0],
            'y': struct.unpack('f', point_bytes[4:8])[0],
            'z': struct.unpack('f', point_bytes[8:12])[0],
            'id': struct.unpack('H', point_bytes[13:15])[0],  # uint16
            'rcs': struct.unpack('f', point_bytes[15:19])[0],
            'vx': struct.unpack('f', point_bytes[19:23])[0],
            'vy': struct.unpack('f', point_bytes[23:27])[0],
        }
        points.append(point)
    
    return points

def plot_simple_xy(points, figsize=(10, 8)):
    """
    绘制简化的X-Y平面点云图
    
    参数:
        points: 点云数据列表
        figsize: 图形大小
    """
    if not points:
        print("没有点云数据可绘制")
        return None
    
    # 提取坐标
    x_coords = [p['x'] for p in points]
    y_coords = [p['y'] for p in points]
    
    # 使用点ID作为颜色
    point_ids = [p['id'] for p in points]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制散点图
    scatter = ax.scatter(x_coords, y_coords, c=point_ids, cmap=plt.cm.tab20, 
                         s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    
    # 设置图形属性
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_title('雷达点云X-Y平面分布')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='datalim')
    
    # 添加雷达位置标记
    ax.plot(0, 0, 'r^', markersize=12, label='雷达位置')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('点ID')
    
    # 添加统计信息
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    total_points = len(points)
    
    stats_text = f"总点数: {total_points}\n"
    stats_text += f"X范围: [{x_min:.2f}, {x_max:.2f}] m\n"
    stats_text += f"Y范围: [{y_min:.2f}, {y_max:.2f}] m"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加图例
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图形
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "radar_xy_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图形已保存到: {output_path}")
    
    plt.show()
    
    return fig

def main():
    # 打开PCD文件
    file_path = "/home/radardepth/data/nuscenes/samples/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178.pcd"
    
    try:
        with open(file_path, "rb") as f:
            h = f.read()
        
        # 找到DATA binary后的数据起始位置
        idx = h.find(b"DATA binary")
        if idx == -1:
            print("错误: 未找到'DATA binary'标记")
            return
        
        pos = idx + len(b"DATA binary")
        while pos < len(h) and h[pos] in (10, 13):  # 跳过换行符
            pos += 1
        
        data = h[pos:]
        
        # 解析点云数据，不依赖表头中的点数
        points = parse_pcd_binary_data(data)
        
        if not points:
            print("没有解析到点云数据")
            return
        
        # 打印基本信息
        print(f"\n实际解析到 {len(points)} 个点")
        if points:
            print("\n第一个点示例:")
            first_point = points[0]
            print(f"  x={first_point['x']:.4f}")
            print(f"  y={first_point['y']:.4f}")
            print(f"  z={first_point['z']:.4f}")
            print(f"  id={first_point['id']}")
            print(f"  rcs={first_point['rcs']:.4f}")
        
        # 绘图
        print("\n=== 绘制X-Y平面点云图 ===")
        
        # 检查matplotlib是否可用
        try:
            import matplotlib
            print("开始绘图...")
            
            # 绘制简化的X-Y平面图
            fig = plot_simple_xy(points)
            
        except ImportError as e:
            print(f"Matplotlib导入错误: {e}")
            print("请安装matplotlib: pip install matplotlib")
            return
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except Exception as e:
        print(f"解析文件时发生错误: {e}")

if __name__ == "__main__":
    main()
