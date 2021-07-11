#!/usr/bin/env python3
# coding=utf-8
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

#如果要增加新的测试模块请在下面的modules后面添加一个名称
#注意：程序输出测试结果的顺序一定要和下面的名称顺序一样！！
#如果模块数量大于colors列表中颜色个数，请在colors中添加一个
#新的颜色，各种颜色名称请参考 https://matplotlib.org/3.1.0/gallery/color/named_colors.html
#添加成功后请在当前目录下运行(make完成后)
#python run_and_draw.py
modules = ["ompp", "reload", "remove", "allpipe", "pipe_1", "dlt"]

colors = ["blue", "orange", "green", "red", "purple", 
            "brown", "pink", "gray", "olive", "cyan"]

module_data = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stencil测试脚本\n运行样例 python run_and_draw.py --n 1000 --N 21000 --interval 1000 --T 10000",
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--n",
        default = 10000,
        type = int,
        help = "问题规模大小N的最小值，即从N等于多少开始测试, 默认从1000开始"
    )
    parser.add_argument(
        "--N",
        default = 80000,
        type = int,
        help = "问题规模大小N的最大值，即测N一直到多少为止, 默认直到20000"
    )
    parser.add_argument(
        "--interval",
        default = 5000,
        type = int,
        help = "从n-N,间隔多少测一次，默认为1000, 即1000, 2000, 3000......20000"
    )
    parser.add_argument(
        "--T",
        default = 5000,
        type = int,
        help = "Stencil运行的时间步长，默认为10000步"
    )
    args = parser.parse_args()
    for module in modules:
        module_data[module] = []

    for i in range(args.n, args.N, args.interval):
        result = os.popen("./exe_1d3p {} {} 5000 1".format(i, args.T))
        res = result.read()
        moduel_index = 0
        for line in res.splitlines():
            if "=" not in line:
                continue
            line = line.split("=")
            module_data[modules[moduel_index]].append(float(line[1]))
            moduel_index = moduel_index + 1

    assert(len(modules) < len(colors))
    x = np.arange(args.n, args.N, args.interval)
    color_index = 0
    lines = []
    for module in modules:
        line,  = plt.plot(x, module_data[module], color = colors[color_index]) 
        lines.append(line)
        color_index = color_index + 1
    
    plt.legend(lines, modules, loc = "upper right")

    plt.xlabel("N")
    plt.ylabel("Stencil/s")
    plt.savefig("result.png")
    plt.show()

