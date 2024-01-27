# 项目名称

多进程通信和一致性算法

## 项目简介

该项目实现了多进程通信和一致性算法。它使用了多进程模块（`multiprocessing`）和相关的类（`Process`、`Queue`、`Manager`、`Barrier`）来实现多个进程之间的通信和协调。同时，项目还使用了`networkx`和`matplotlib`库来绘制通信拓扑图和代理数值的变化趋势。

## 功能特性

- 根据设定的通信拓扑连接多个进程之间的通信关系
- 每个进程具有独立的数值，并通过一致性算法实现数值的一致性
- 绘制通信拓扑图，展示各个进程之间的通信连接
- 绘制代理数值的变化趋势图，展示一致性算法的收敛过程


## 项目结论

- 验证了多进程和单进程之间当每个agent具有不同运行时间的时候，多进程的运行时间更短
-  进一步验证了不管是多进程还是单进程，异步算法的运行时间都比同步算法的运行时间一版要长一些。



## 代码示例

```python
# 导入所需的库
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

# 其他代码...
# 省略了部分代码，完整代码请参考上方代码段

if __name__ == "__main__":
    multiprocess_queue()
    # singleprocess()
```

## 安装和运行

1. 克隆项目代码到本地：
   ```
   git clone https://github.com/your/repo.git
   ```

2. 安装依赖库：
   ```
   pip install -r requirements.txt
   ```

3. 运行项目：
   ```
   python main.py
   ```

## 贡献者

- Dr.Junfeng Chen


