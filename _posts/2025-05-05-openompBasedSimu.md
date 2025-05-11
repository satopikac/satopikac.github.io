---
layout: post
title: "基于 OpenMP 的多线程演化博弈模拟代码"
date:   2025-05-05
tags: [geek]
comments: true
author: satopikac
---
# 基于 OpenMP 的多线程演化博弈模拟代码

本项目是一个使用 **C++17** 和 **OpenMP** 实现的并行演化博弈模拟程序，用于研究不同网络结构（规则网络与无标度网络）下囚徒困境（Prisoner's Dilemma, PD）和雪堆博弈（Snowdrift Game, SG）中合作行为的演化过程。项目请见:[github](https://github.com/satopikac/Cooperation-In-Scale-Free-Network-and-Regular-Network).论文请见[点击查看 PDF](/assets/pdf/output.pdf)

## 🧠 核心功能概述

该程序主要完成以下任务：

- 构建两种类型的网络：**规则网络（Regular Network）** 和 **无标度网络（Scale-Free Network）**
- 在这些网络上进行 **演化博弈模拟**
- 使用 **OpenMP 多线程加速** 进行参数扫描
- 输出每组参数下的平均合作比例到 CSV 文件中

---

## 📁 程序结构概览

### 主要类

| 类名               | 功能                                   |
| ------------------ | -------------------------------------- |
| `Network`        | 负责构建网络结构（规则图 / 无标度图）  |
| `GameSimulation` | 实现博弈逻辑、策略更新、合作比例计算等 |
| `main()` 函数    | 控制流程、调用模拟、输出结果           |

---

## ⚙️ 模拟参数设置

```cpp
int allgenerations = 4000;      // 总代数
int thetransient = 3000;        // 预热期（不记录数据）
const int N = 400;              // 网络节点数
const int repeats = 50;         // 每个参数重复运行次数

// 囚徒困境参数范围
const double pd_b_start = 1.0;
const double pd_b_end = 2.0;
const int pd_steps = 21;

// 雪堆博弈参数范围
const double sg_b_start = 0.0;
const double sg_b_end = 1.0;
const int sg_steps = 11;
```

---

## 🧬 网络类 (`Network`)

### 支持的网络类型

#### 1. 规则网络 (`buildRegular`)

每个节点与其最近的 `z/2` 个邻居相连（双向），形成一个环形结构。

#### 2. 无标度网络 (`buildScaleFree`)

使用 BA 模型：

- 初始有 `m0` 个完全连接的节点
- 新节点以概率正比于现有节点的度接入已有节点，实现“富者愈富”

---

## 🎮 博弈模拟类 (`GameSimulation`)

### 初始化策略

- 一半节点初始化为合作者（Cooperator，策略值为1）
- 另一半为背叛者（Defector，策略值为0）

### 博弈类型支持

#### 1. 囚徒困境 (PD)

收益矩阵如下：

| Player A \ B  | Cooperate (1) | Defect (0) |
| ------------- | ------------- | ---------- |
| Cooperate (1) | 1             | 0          |
| Defect (0)    | b             | 0          |

#### 2. 雪堆博弈 (SG)

收益矩阵如下：

| Player A \ B  | Cooperate (1) | Defect (0) |
| ------------- | ------------- | ---------- |
| Cooperate (1) | β - 0.5      | β - 1     |
| Defect (0)    | β            | 0          |

其中

$\beta=\frac{1}{2}(\frac{1}{b}+1)$

### 策略更新规则

采用 **模仿最优邻域策略**：

- 如果随机选择一个邻居的收益大于当前节点，则有一定概率模仿其策略。
- 更新概率公式为：

$$
W = \frac{P_j - P_i}{b \cdot k}
$$

其中 $k$ 是两个节点中较大的度数。

---

## 🔀 并行参数扫描函数 (`parameterSweep`)

该函数对给定参数范围进行多次模拟并取平均合作比例。

### 输入参数

| 参数名          | 含义                              |
| --------------- | --------------------------------- |
| networkType     | `"regular"` 或 `"scale-free"` |
| gameType        | `"PD"` 或 `"SG"`              |
| param_start/end | 参数起止值                        |
| steps           | 步数                              |
| N, z            | 网络大小和平均度                  |
| repeats         | 每个参数重复次数                  |

### 并行化方式

- 使用 `#pragma omp parallel for schedule(dynamic)` 对参数循环并行
- 内部使用 `reduction(+:coop_sum)` 对每次重复的结果求和

---

## 🧾 主函数 (`main`)

主函数会遍历多个不同的平均度 `z` 值（如 4, 8, 12, 16, 32），分别运行：

- 囚徒困境在规则网络和无标度网络上的模拟
- 雪堆博弈在规则网络和无标度网络上的模拟

### 输出文件格式

生成两个 CSV 文件：

- `pd_results_z=*.csv`
- `sg_results_z=*.csv`

内容格式如下：

```
b,regular,scale-free
1.0,0.45,0.56
...
```

表示在给定的 `b` (对于SG是r)值下，规则网络和无标度网络中的平均合作比例。

---

## 🧰 编译与运行方法

### 编译命令（需支持 C++17 和 OpenMP）：

```bash
g++ -std=c++17 -O3 -fopenmp parallel.cpp -o parallel
```

### 运行程序：

```bash
./parallel
```

windows环境下还可使用Visual Studio打开github中CMake项目进行编译运行。

---

## ✅ 示例输出

运行后会在当前目录生成如下文件（每个 `z` 值一组）：

```
pd_results_z=4.csv
sg_results_z=4.csv
pd_results_z=8.csv
sg_results_z=8.csv
...
```

每一行代表在某个收益参数 `b` 下的合作比例。

---

## 📈 应用场景

此程序可用于研究：

- 不同网络结构对合作演化的促进作用
- 博弈收益参数对合作频率的影响
- 度分布差异（规则 vs 无标度）对演化动力学的影响

---

## 📌 注意事项

- 确保编译器支持 OpenMP（如 GCC >= 9）
- 可根据需要调整 `N`, `allgenerations`, `repeats` 等参数

## 📝 源码

```c++
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <numeric>
#include <omp.h> // OpenMP 
using namespace std;

// 每个线程都有自己的随机数生成器
thread_local mt19937 thread_rng(chrono::steady_clock::now().time_since_epoch().count() ^ omp_get_thread_num());


//模拟参数
int allgenerations = 4000;
int thetransient = 3000;
const int N = 400;
//const int z = 4;
const int repeats = 50;

const double pd_b_start = 1.0;
const double pd_b_end = 2.0;
const int pd_steps = 21;

const double sg_b_start = 0.0;
const double sg_b_end = 1.0;
const int sg_steps = 11;

// network class
class Network {
public:
    int N; // Number of nodes节点数
    int z; // Average degree平均度
    vector<vector<int>> adj; // Adjacency list邻接表
    Network(int n, int k) : N(n), z(k), adj(n) {}
  
    void buildRegular() {
        for (int i = 0; i < N; ++i) {
            for (int j = 1; j <= z/2; ++j) {
                int neighbor = (i + j) % N;
                adj[i].push_back(neighbor);
                adj[neighbor].push_back(i);
            }
        }
    }

    void buildScaleFree(int m0 = 5) {
        for (int i = 0; i < m0; ++i) {
            for (int j = i+1; j < m0; ++j) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
        for (int i = m0; i < N; ++i) {
            vector<double> probs;
            int total_degree = 0;
            for (int j = 0; j < i; ++j) {
                total_degree += adj[j].size();
            }
            for (int j = 0; j < i; ++j) {
                probs.push_back(adj[j].size() / (double)total_degree);
            }
            discrete_distribution<int> dist(probs.begin(), probs.end());
            for (int j = 0; j < z/2; ++j) {
                int target = dist(thread_rng);
                if (find(adj[i].begin(), adj[i].end(), target) == adj[i].end()) {
                    adj[i].push_back(target);
                    adj[target].push_back(i);
                }
            }
        }
    }

    int degree(int node) {
        return adj[node].size();
    }
};

// Game simulation class
class GameSimulation {
public:
    Network& network;
    string gameType;
    double b;
    double r;
    vector<int> strategies;
    vector<double> payoffs;
    GameSimulation(Network& net, string type, double param_b, double param_r = 0) 
        : network(net), gameType(type), b(param_b), r(param_r), 
          strategies(net.N), payoffs(net.N) {}

    void initializeStrategies() {
        vector<int> indices(network.N);
        for (int i = 0; i < network.N; ++i) indices[i] = i;
        shuffle(indices.begin(), indices.end(), thread_rng);
        int half_N = network.N / 2;
        for (int i = 0; i < half_N; ++i) strategies[indices[i]] = 0; // Defector
        for (int i = half_N; i < network.N; ++i) strategies[indices[i]] = 1; // Cooperator
    }

    void playRound() {
        fill(payoffs.begin(), payoffs.end(), 0.0);
        for (int i = 0; i < network.N; ++i) {
            for (int j : network.adj[i]) {
                if (j > i) continue;
                double payoff_i, payoff_j;
                if (gameType == "PD") {
                    if (strategies[i] == 1 && strategies[j] == 1) { payoff_i = payoff_j = 1; }
                    else if (strategies[i] == 1 && strategies[j] == 0) { payoff_i = 0; payoff_j = b; }
                    else if (strategies[i] == 0 && strategies[j] == 1) { payoff_i = b; payoff_j = 0; }
                    else { payoff_i = payoff_j = 0; }
                } else { // SG
                    double beta = (1/b + 1)*0.5;
                    if (strategies[i] == 1 && strategies[j] == 1) { payoff_i = payoff_j = beta - 0.5; }
                    else if (strategies[i] == 1 && strategies[j] == 0) { payoff_i = beta - 1; payoff_j = beta; }
                    else if (strategies[i] == 0 && strategies[j] == 1) { payoff_i = beta; payoff_j = beta - 1; }
                    else { payoff_i = payoff_j = 0; }
                }
                payoffs[i] += payoff_i;
                payoffs[j] += payoff_j;
            }
        }
    }

    void updateStrategies() {
        uniform_real_distribution<double> prob(0.0, 1.0);
        for (int i = 0; i < network.N; ++i) {
            if (network.adj[i].empty()) continue;
            uniform_int_distribution<int> neighbor_dist(0, network.adj[i].size()-1);
            int j = network.adj[i][neighbor_dist(thread_rng)];
            if (payoffs[j] > payoffs[i]) {
                double D = b;
                double k = max(network.degree(i), network.degree(j));
                double W = (payoffs[j] - payoffs[i]) / (D * k);
                if (prob(thread_rng) < W) {
                    strategies[i] = strategies[j];
                }
            }
        }
    }

    double cooperatorFraction() {
        int count = accumulate(strategies.begin(), strategies.end(), 0);
        return count / (double)network.N;
    }

    double runSimulation(int generations = allgenerations, int transient = thetransient) {
        initializeStrategies();
        vector<double> coop_fractions;
        for (int g = 1; g <= generations; ++g) {
            playRound();
            updateStrategies();
            if (g > transient) {
                coop_fractions.push_back(cooperatorFraction());
            }
        }
        return accumulate(coop_fractions.begin(), coop_fractions.end(), 0.0) / coop_fractions.size();
    }
};

// Parallelized parameter sweep using OpenMP
vector<vector<double>> parameterSweep(string networkType, string gameType, 
                                     double param_start, double param_end, int steps, 
                                     int N = 1000, int z = 4, int repeats = 10) {
    vector<vector<double>> results(steps, vector<double>(2));
    double step_size = (param_end - param_start) / (steps - 1);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < steps; ++i) {
        double param = param_start + i * step_size;
        double coop_sum = 0.0;

        #pragma omp parallel for reduction(+:coop_sum)
        for (int r = 0; r < repeats; ++r) {
            Network net(N, z);
            if (networkType == "regular") {
                net.buildRegular();
            } else {
                net.buildScaleFree();
            }
            GameSimulation sim(net, gameType, param, param);
            coop_sum += sim.runSimulation();
        }

        results[i][0] = param;
        results[i][1] = coop_sum / repeats;
    }
    return results;
}

int main() {
    vector<int> Z{4,8,12,16,32};
for(auto z:Z){
    cout<<"now begin z="<<z<<" simulation"<<endl;
    auto pd_regular = parameterSweep("regular", "PD", pd_b_start, pd_b_end, pd_steps, N, z, repeats);
    auto pd_sf = parameterSweep("scale-free", "PD", pd_b_start, pd_b_end, pd_steps, N, z, repeats);

    auto sg_regular = parameterSweep("regular", "SG", sg_b_start, sg_b_end, sg_steps, N, z, repeats);
    auto sg_sf = parameterSweep("scale-free", "SG", sg_b_start, sg_b_end, sg_steps, N, z, repeats);

    cout << "begin cal" << endl;

    string pd_filename = "pd_results_z=" + std::to_string(z) + ".csv";
    ofstream out_pd(pd_filename);
    out_pd << "b,regular,scale-free\n";
    for (int i = 0; i < pd_steps; ++i) {
        out_pd << pd_regular[i][0] << "," << pd_regular[i][1] << "," << pd_sf[i][1] << "\n";
    }
    out_pd.close();

    string sg_filename = "sg_results_z=" + std::to_string(z) + ".csv";
    ofstream out_sg(sg_filename);

    out_sg << "b,regular,scale-free\n";
    for (int i = 0; i < sg_steps; ++i) {
        if(i==0)continue;
        out_sg << sg_regular[i][0] << "," << sg_regular[i][1] << "," << sg_sf[i][1] << "\n";
    }
    out_sg.close();

    cout << "z="<<z<<" Simulations completed. Results saved to pd_results.csv and sg_results.csv\n";
  
}
return 0;
}


//使用C++ 17标准编译
// g++ -std=c++17 -O3 -fopenmp parallel.cpp -o parallel
// ./parallel
```

---
