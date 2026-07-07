import numpy as np
import matplotlib.pyplot as plt
import math

def plot_decay_curve():
    # 参数设置
    T = 10          # 当前时间步
    recent_k = 10   # 回溯多少步
    decay_lambda = 0.3 # 你的衰减系数
    
    # 生成时间轴 t
    t = np.arange(T - recent_k + 1, T + 1)
    
    # 你的公式
    weights = np.exp(-decay_lambda * (T - t))
    
    # --- 开始绘图 ---
    plt.figure(figsize=(8, 4), dpi=150)
    
    # 1. 画连续的曲线 (作为背景参考)
    x_cont = np.linspace(T - recent_k + 1, T, 100)
    y_cont = np.exp(-decay_lambda * (T - x_cont))
    plt.plot(x_cont, y_cont, 'r--', alpha=0.5, label=r'Decay Function $e^{-\lambda(T-t)}$')
    
    # 2. 画离散的权重 (Stem plot / 棒棒糖图)
    # 这是最适合表示离散 Visit 权重的画法
    markerline, stemlines, baseline = plt.stem(t, weights, linefmt='b-', markerfmt='bo', basefmt=" ")
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, 'markersize', 8)
    
    # 3. 标注最新的 Visit
    plt.annotate('Current Visit (T)\nWeight = 1.0', xy=(T, 1), xytext=(T-2.5, 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 4. 标注旧的 Visit
    plt.annotate(f'Old Visit\nWeight ≈ {weights[0]:.2f}', xy=(t[0], weights[0]), xytext=(t[0]+0.5, weights[0]+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 装饰
    plt.title(f'Acute Pathway Temporal Decay ($\lambda={decay_lambda}$)', fontsize=12)
    plt.xlabel('Time Step (Visit Sequence)', fontsize=10)
    plt.ylabel('Attention Weight $w$', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('time.pdf', dpi=300)

if __name__ == "__main__":
    plot_decay_curve()
