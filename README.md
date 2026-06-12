# 熵正则化最优运输 — Sinkhorn 算法实现与实验

最优化课程项目：从 Monge 土方搬运问题到 Cuturi (2013) 的 Sinkhorn 迭代，实现 5 种求解算法并通过 8 组实验验证理论预言。

## 快速开始

```bash
# 安装依赖
pip install numpy matplotlib Pillow

# 一键运行全部实验（~60 秒）
python main.py

# 只运行指定实验
python main.py --only e1,e5,e8
```

图表自动保存至 `figures/` 目录。

## 项目结构

```
├── main.py                  # 主脚本
├── solvers/                 # 5 种求解器
│   ├── sinkhorn.py          # A1: Sinkhorn 迭代（2 行核心循环）
│   ├── dual_gradient.py     # A2: 对偶梯度上升
│   ├── log_sinkhorn.py      # A3: log 域稳定 Sinkhorn
│   ├── admm_lp.py           # A4: ADMM 解 LP (ε=0)
│   └── eps_scaling.py       # A5: ε-scaling 同伦
├── data/                    # 数据生成与解析真解
├── utils/                   # 指标计算与可视化
├── experiments/             # 8 个实验脚本 (run_e1.py ~ run_e8.py)
└── figures/                 # 输出图表 (9 张 PNG)
```

## 算法一览

| 算法 | 核心思想 | 收敛速度 | 每步开销 |
|:---|:---|:---:|:---:|
| A1 Sinkhorn | 对偶分块坐标上升 | 线性 | $O(n^2)$，2 次矩阵-向量乘 |
| A2 对偶梯度 | 光滑凹函数梯度法 | $O(1/k)$ | $O(n^2)$，重算 $\Pi$ |
| A3 log-Sinkhorn | log-sum-exp 稳定化 | 同 A1 | $O(n^2)$，常数 ×2-3 |
| A4 ADMM-LP | 增广拉格朗日交替极小 | $O(1/k)$ | $O(n^2)$，仿射投影 |
| A5 ε-scaling | 同伦 warm start | 框架性加速 | 同 A3 |

**核心约束**：仅依赖 `numpy`，所有算法主体为手写循环，零优化求解包。

## 实验概览

| 实验 | 内容 | 输出图 |
|:---:|:---|:---|
| E1 | ε 对运输方案的影响（热力图） | `fig_e1_heatmaps.png` |
| E2 | ε 对收敛速度的影响 | `fig_e2_convergence_vs_eps.png` |
| E3 | ε-scaling 同伦加速 | `fig_e3_eps_scaling.png` |
| E4 | 数值稳定性：A1 崩溃 vs A3 稳定 | `fig_e4_stability.png` |
| E5 | 四种算法收敛对比 | `fig_e5_algorithm_comparison.png` |
| E6 | ADMM 罚因子 ρ 敏感性 | `fig_e6_admm_rho.png` |
| E7 | 规模扩展性 $O(n^2)$ | `fig_e7_scalability.png` |
| E8 | 图像色彩迁移 | `fig_e8_color_transfer.png` |

## 文档

- `项目报告_成员1成员2_新选题_最优运输Sinkhorn.md` — 问题建模与算法设计
- `实验报告_成员3成员4_代码实现与实验分析.md` — 代码实现、运行指南与实验分析

## 参考文献

- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *NeurIPS*.
- Peyré, G. & Cuturi, M. (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning*.
- Franklin, J. & Lorenz, J. (1989). On the scaling of multidimensional matrices. *Linear Algebra and its Applications*.
