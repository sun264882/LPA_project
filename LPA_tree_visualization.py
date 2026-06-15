"""
用“栅格地图 + 最优前驱树 + OPEN 队列”直观演示 LPA*。

运行：
    python LPA_tree_visualization.py

无界面环境生成静态结果：
    python LPA_tree_visualization.py --save-only

交互按键：
    Space  暂停/继续
    Right  暂停时单步执行
    Q      关闭窗口
"""

import argparse
import heapq
import math
import sys
import textwrap

import matplotlib

if "--save-only" in sys.argv:
    matplotlib.use("Agg", force=True)
else:
    # PyCharm 的 Scientific/Plots 工具窗口可能把 pyplot 接管为静态显示。
    # 强制使用 TkAgg，让动画始终在独立 GUI 窗口中刷新。
    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError as exc:
        raise RuntimeError(
            "动态模式需要 Tk。请确认当前 Python 解释器能够 import tkinter，"
            "或者使用 --save-only 生成静态图片。"
        ) from exc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


INF = float("inf")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class GridMap:
    """只允许上下左右移动、每条边代价为 1 的二维栅格地图。"""

    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles or [])

    def in_bounds(self, node):
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, node):
        return self.in_bounds(node) and node not in self.obstacles

    def neighbors(self, node):
        if not self.is_free(node):
            return []

        x, y = node
        # 固定顺序只用于让演示结果容易复现。
        candidates = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        return [(nxt, 1.0) for nxt in candidates if self.is_free(nxt)]

    def adjacent_cells(self, node):
        x, y = node
        candidates = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        return [nxt for nxt in candidates if self.in_bounds(nxt)]


class VisualLPAStar:
    """
    带状态记录的 LPA*。

    g:   当前保存的起点到节点的代价。
    rhs: 根据当前相邻节点计算的一步前瞻代价。
    parent: 产生 rhs 最小值的前驱，仅用于绘制“最优前驱树”。
    """

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal

        self.g = {}
        self.rhs = {start: 0.0}
        self.parent = {}

        self.open_heap = []
        self.open_dict = {}
        self.insert(start)

    def get_g(self, node):
        return self.g.get(node, INF)

    def get_rhs(self, node):
        return self.rhs.get(node, INF)

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def key(self, node):
        value = min(self.get_g(node), self.get_rhs(node))
        return value + self.heuristic(node), value

    @staticmethod
    def key_less(left, right):
        return left[0] < right[0] or (
            left[0] == right[0] and left[1] < right[1]
        )

    def insert(self, node):
        key = self.key(node)
        self.open_dict[node] = key
        heapq.heappush(self.open_heap, (key, node))

    def remove(self, node):
        self.open_dict.pop(node, None)

    def top_key(self):
        while self.open_heap:
            key, node = self.open_heap[0]
            if self.open_dict.get(node) == key:
                return key
            heapq.heappop(self.open_heap)
        return INF, INF

    def pop(self):
        while self.open_heap:
            key, node = heapq.heappop(self.open_heap)
            if self.open_dict.get(node) == key:
                del self.open_dict[node]
                return node
        return None

    def update_vertex(self, node):
        if not self.grid.in_bounds(node):
            return

        if not self.grid.is_free(node):
            self.g[node] = INF
            self.rhs[node] = INF
            self.parent.pop(node, None)
            self.remove(node)
            return

        if node != self.start:
            choices = [
                (self.get_g(predecessor) + cost, predecessor)
                for predecessor, cost in self.grid.neighbors(node)
            ]
            best_value, best_parent = min(choices, default=(INF, None))
            self.rhs[node] = best_value

            if best_parent is None or math.isinf(best_value):
                self.parent.pop(node, None)
            else:
                self.parent[node] = best_parent

        self.remove(node)
        if self.get_g(node) != self.get_rhs(node):
            self.insert(node)

    def finished(self):
        return (
            not self.key_less(self.top_key(), self.key(self.goal))
            and self.get_g(self.goal) == self.get_rhs(self.goal)
        )

    def step(self):
        """执行一次队列弹出，并返回可视化需要的解释信息。"""
        if self.finished():
            return None

        node = self.pop()
        if node is None:
            return None

        old_g = self.get_g(node)
        old_rhs = self.get_rhs(node)

        if old_g > old_rhs:
            self.g[node] = old_rhs
            action = (
                f"{node}: g > rhs，发现更短/新可达路径，"
                f"令 g = rhs = {format_number(old_rhs)}"
            )
            for successor, _ in self.grid.neighbors(node):
                self.update_vertex(successor)
        else:
            self.g[node] = INF
            action = (
                f"{node}: g < rhs，旧代价已经失效，"
                "先令 g = ∞，再向后继传播变化"
            )
            self.update_vertex(node)
            for successor, _ in self.grid.neighbors(node):
                self.update_vertex(successor)

        return {
            "node": node,
            "old_g": old_g,
            "old_rhs": old_rhs,
            "action": action,
        }

    def add_obstacle(self, cell):
        """加入障碍并只更新受该变化直接影响的局部节点。"""
        self.grid.obstacles.add(cell)
        self.g[cell] = INF
        self.rhs[cell] = INF
        self.parent.pop(cell, None)
        self.remove(cell)

        for node in self.grid.adjacent_cells(cell):
            self.update_vertex(node)

    def extract_path(self):
        """从终点沿 g 值最小的前驱回溯到起点。"""
        if math.isinf(self.get_g(self.goal)):
            return []

        path = [self.goal]
        current = self.goal
        visited = {current}

        while current != self.start:
            choices = [
                (self.get_g(predecessor) + cost, predecessor)
                for predecessor, cost in self.grid.neighbors(current)
            ]
            _, predecessor = min(choices, default=(INF, None))
            if predecessor is None or predecessor in visited:
                return []

            path.append(predecessor)
            visited.add(predecessor)
            current = predecessor

        path.reverse()
        return path

    def discovered_nodes(self):
        nodes = set()
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                node = (x, y)
                if not math.isinf(min(self.get_g(node), self.get_rhs(node))):
                    nodes.add(node)
        return nodes

    def node_state(self, node):
        g_value = self.get_g(node)
        rhs_value = self.get_rhs(node)
        if g_value == rhs_value:
            return "consistent"
        if g_value > rhs_value:
            return "improve"
        return "invalid"


def format_number(value):
    if math.isinf(value):
        return "∞"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def format_key(key):
    return f"({format_number(key[0])}, {format_number(key[1])})"


class AnimationControl:
    def __init__(self, figure):
        self.paused = False
        self.single_step = False
        self.closed = False
        figure.canvas.mpl_connect("key_press_event", self.on_key)
        figure.canvas.mpl_connect("close_event", self.on_close)

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "right" and self.paused:
            self.single_step = True
        elif event.key in {"q", "escape"}:
            self.closed = True
            plt.close(event.canvas.figure)

    def on_close(self, _event):
        self.closed = True

    def wait_until_allowed(self):
        while self.paused and not self.single_step and not self.closed:
            plt.pause(0.05)
        self.single_step = False


class LPAVisualizer:
    STATE_COLORS = {
        "consistent": "#B7E4C7",
        "improve": "#90CAF9",
        "invalid": "#FFAB91",
    }

    def __init__(self, planner, save_only=False):
        self.planner = planner
        self.grid = planner.grid
        self.save_only = save_only

        if not save_only:
            plt.ion()

        self.figure = plt.figure(figsize=(17, 8.5))
        layout = self.figure.add_gridspec(
            2,
            3,
            width_ratios=[1.0, 1.75, 0.75],
            height_ratios=[1.05, 0.95],
            wspace=0.22,
            hspace=0.20,
        )
        self.map_ax = self.figure.add_subplot(layout[:, 0])
        self.tree_ax = self.figure.add_subplot(layout[:, 1])
        self.queue_ax = self.figure.add_subplot(layout[0, 2])
        self.info_ax = self.figure.add_subplot(layout[1, 2])

        if not save_only:
            # 必须在算法循环之前显示窗口，否则 PyCharm 可能只显示最终一帧。
            self.figure.canvas.manager.set_window_title("LPA* 动态树状可视化")
            plt.show(block=False)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            plt.pause(0.1)

    def draw(self, phase, step_number, current=None, expanded=None, path=None, message=""):
        self.draw_grid(current, expanded or set(), path or [])
        self.draw_tree(current, path or [])
        self.draw_queue()
        self.draw_information(phase, step_number, current, path or [], message)
        self.figure.suptitle(
            "LPA* 直观演示：地图状态、最优前驱树与 OPEN 队列",
            fontsize=17,
            fontweight="bold",
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def draw_grid(self, current, expanded, path):
        ax = self.map_ax
        ax.clear()
        path_set = set(path)
        open_nodes = set(self.planner.open_dict)

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                node = (x, y)
                color = "white"
                if node in expanded:
                    color = "#E8F5E9"
                if node in open_nodes:
                    color = "#E3F2FD"
                if node in path_set:
                    color = "#FFF59D"
                if node == current:
                    color = "#CE93D8"
                if node in self.grid.obstacles:
                    color = "#37474F"

                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="#78909C",
                        linewidth=1,
                    )
                )

                if (
                    node not in self.grid.obstacles
                    and node not in {self.planner.start, self.planner.goal}
                ):
                    g_text = format_number(self.planner.get_g(node))
                    rhs_text = format_number(self.planner.get_rhs(node))
                    if g_text != "∞" or rhs_text != "∞":
                        ax.text(
                            x,
                            y + 0.22,
                            f"g={g_text}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                        ax.text(
                            x,
                            y - 0.18,
                            f"r={rhs_text}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="#F57F17", linewidth=3, zorder=4)

        sx, sy = self.planner.start
        gx, gy = self.planner.goal
        ax.text(sx, sy, "S", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(gx, gy, "G", ha="center", va="center", fontsize=15, fontweight="bold")

        ax.set_xlim(-0.5, self.grid.width - 0.5)
        ax.set_ylim(self.grid.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.grid.width))
        ax.set_yticks(range(self.grid.height))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("栅格地图（黄色才是最终路径）", fontsize=13, fontweight="bold")

    def tree_positions(self, nodes):
        layers = {}
        for node in nodes:
            depth_value = min(self.planner.get_g(node), self.planner.get_rhs(node))
            depth = int(depth_value)
            layers.setdefault(depth, []).append(node)

        positions = {}
        for depth, layer_nodes in sorted(layers.items()):
            layer_nodes.sort(key=lambda node: (node[1], node[0]))
            if len(layer_nodes) == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.95, 0.05, len(layer_nodes))
            for node, y_position in zip(layer_nodes, y_positions):
                positions[node] = (depth, float(y_position))
        return positions

    def draw_tree(self, current, path):
        ax = self.tree_ax
        ax.clear()
        nodes = self.planner.discovered_nodes()
        positions = self.tree_positions(nodes)
        path_edges = set(zip(path, path[1:]))
        path_edges |= {(right, left) for left, right in path_edges}

        for node in nodes:
            parent = self.planner.parent.get(node)
            if parent not in positions or node not in positions:
                continue

            start = positions[parent]
            end = positions[node]
            is_path_edge = (parent, node) in path_edges
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=2.4 if is_path_edge else 0.8,
                color="#F57F17" if is_path_edge else "#90A4AE",
                alpha=0.95 if is_path_edge else 0.75,
                zorder=1,
            )
            ax.add_patch(arrow)

        for node, (x_position, y_position) in positions.items():
            state = self.planner.node_state(node)
            color = self.STATE_COLORS[state]
            edge_color = "#455A64"
            line_width = 1.0

            if node in self.planner.open_dict:
                edge_color = "#1565C0"
                line_width = 2.5
            if node == self.planner.goal:
                edge_color = "#C62828"
                line_width = 2.8
            if node == current:
                color = "#CE93D8"
                edge_color = "#6A1B9A"
                line_width = 3.0

            ax.scatter(
                [x_position],
                [y_position],
                s=620,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=line_width,
                zorder=3,
            )
            ax.text(
                x_position,
                y_position,
                (
                    f"{node}\n"
                    f"{format_number(self.planner.get_g(node))}/"
                    f"{format_number(self.planner.get_rhs(node))}"
                ),
                ha="center",
                va="center",
                fontsize=6.6,
                zorder=4,
            )

        max_depth = max((position[0] for position in positions.values()), default=1)
        ax.set_xlim(-0.8, max_depth + 0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([])
        ax.set_xticks(range(max_depth + 1))
        ax.set_xlabel("树的层数 = 当前从起点出发的代价")
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_title(
            "当前最优前驱树（节点内为 坐标 和 g/rhs）",
            fontsize=13,
            fontweight="bold",
        )

    def draw_queue(self):
        ax = self.queue_ax
        ax.clear()
        ax.axis("off")
        ax.set_title("OPEN 优先队列（前 12 项）", fontsize=12, fontweight="bold")

        entries = sorted(
            self.planner.open_dict.items(),
            key=lambda item: (item[1], item[0]),
        )
        lines = []
        for index, (node, key) in enumerate(entries[:12], start=1):
            state = self.planner.node_state(node)
            symbol = {"consistent": "=", "improve": "↓", "invalid": "↑"}[state]
            lines.append(f"{index:>2}. {node!s:<7} k={format_key(key):<9} {symbol}")

        if not lines:
            lines = ["OPEN 为空"]

        ax.text(
            0.02,
            0.96,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            family="monospace",
            fontsize=9.5,
            linespacing=1.35,
        )

        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#B7E4C7",
                   markeredgecolor="#455A64", markersize=9, label="g = rhs：一致"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9",
                   markeredgecolor="#1565C0", markersize=9, label="g > rhs：等待降低 g"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFAB91",
                   markeredgecolor="#455A64", markersize=9, label="g < rhs：旧代价失效"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#CE93D8",
                   markeredgecolor="#6A1B9A", markersize=9, label="当前处理节点"),
            Line2D([0], [0], color="#F57F17", linewidth=3, label="最终路径树边"),
        ]
        ax.legend(handles=legend_items, loc="lower left", fontsize=7.5)

    def draw_information(self, phase, step_number, current, path, message):
        ax = self.info_ax
        ax.clear()
        ax.axis("off")
        ax.set_title("当前步骤说明", fontsize=12, fontweight="bold")

        goal_g = format_number(self.planner.get_g(self.planner.goal))
        goal_rhs = format_number(self.planner.get_rhs(self.planner.goal))
        wrapped_message = "\n".join(
            textwrap.fill(line, width=24) if line else ""
            for line in message.splitlines()
        )
        text = (
            f"阶段：{phase}\n"
            f"处理次数：{step_number}\n"
            f"当前节点：{current if current is not None else '-'}\n"
            f"终点 g/rhs：{goal_g}/{goal_rhs}\n"
            f"当前路径长度：{max(len(path) - 1, 0) if path else '-'}\n\n"
            f"{wrapped_message}\n\n"
            "读图方法：\n"
            "1. 树边表示节点当前选择的最佳前驱。\n"
            "2. 蓝色边框节点仍在 OPEN 中等待处理。\n"
            "3. 橙色树边和黄色格子才组成最终路径。\n"
            "4. 相邻动画帧的当前节点可以不相邻。\n\n"
            "按键：Space 暂停/继续\n"
            "      Right 暂停时单步\n"
            "      Q 关闭"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            linespacing=1.4,
            wrap=True,
        )

    def save(self, filename):
        self.figure.savefig(filename, dpi=170, bbox_inches="tight")


def create_demo():
    width, height = 9, 6
    start = (0, 2)
    goal = (8, 2)

    # 两道墙各有上下两个通道。初始最短路走上通道，动态障碍出现后改走下通道。
    obstacles = {
        (3, 0), (3, 2), (3, 3), (3, 5),
        (6, 0), (6, 2), (6, 3), (6, 5),
    }
    grid = GridMap(width, height, obstacles)
    return VisualLPAStar(grid, start, goal)


def responsive_pause(seconds, control):
    elapsed = 0.0
    while elapsed < seconds and not control.closed:
        if control.paused:
            plt.pause(0.05)
            continue
        interval = min(0.05, seconds - elapsed)
        plt.pause(interval)
        elapsed += interval


def run_phase(planner, visualizer, control, phase, delay, save_only):
    expanded = set()
    step_number = 0
    last_event = None

    while not planner.finished() and not control.closed:
        if not save_only:
            control.wait_until_allowed()
            if control.closed:
                break

        last_event = planner.step()
        if last_event is None:
            break

        current = last_event["node"]
        expanded.add(current)
        step_number += 1

        if not save_only:
            visualizer.draw(
                phase,
                step_number,
                current=current,
                expanded=expanded,
                message=last_event["action"],
            )
            # 某些 Matplotlib 后端会把 pause(0) 解释为无限事件循环。
            plt.pause(max(delay, 0.001))

    return expanded, step_number, last_event


def run_demo(delay, save_only, output):
    backend = matplotlib.get_backend()
    non_interactive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if not save_only and (
        backend.lower() in non_interactive_backends
        or "inline" in backend.lower()
        or "interagg" in backend.lower()
    ):
        raise RuntimeError(
            f"当前 Matplotlib 后端为 {backend}，它不支持动态窗口。"
            "请不要使用 --save-only，并在 PyCharm 中关闭 "
            "Settings | Tools | Python Scientific | Show plots in tool window。"
        )

    print(f"Matplotlib 后端: {backend}")
    if not save_only:
        print("动态窗口已启用。Space 暂停/继续，Right 单步，Q 关闭。")

    planner = create_demo()
    visualizer = LPAVisualizer(planner, save_only=save_only)
    control = AnimationControl(visualizer.figure)

    if not save_only:
        visualizer.draw(
            "初始化",
            0,
            message="只有起点 rhs=0，因此起点首先进入 OPEN。",
        )
        responsive_pause(1.0, control)

    expanded, steps, _ = run_phase(
        planner,
        visualizer,
        control,
        "第一次规划：从起点向外建立代价树",
        delay,
        save_only,
    )
    initial_path = planner.extract_path()

    if not initial_path:
        raise RuntimeError("初始地图不存在可行路径。")

    if not save_only and not control.closed:
        visualizer.draw(
            "第一次规划完成",
            steps,
            expanded=expanded,
            path=initial_path,
            message="终点已经一致。黄色路径经过上方通道 (3, 1)。",
        )
        responsive_pause(1.8, control)

    dynamic_obstacle = (3, 1)
    if dynamic_obstacle not in initial_path:
        raise RuntimeError("演示障碍不在初始路径上，请检查地图配置。")

    planner.add_obstacle(dynamic_obstacle)

    if not save_only and not control.closed:
        visualizer.draw(
            "地图变化",
            0,
            path=initial_path,
            message=(
                f"{dynamic_obstacle} 变为障碍。旧黄色路径失效，"
                "LPA* 只把局部不一致节点放回 OPEN。"
            ),
        )
        responsive_pause(1.8, control)

    repair_expanded, repair_steps, _ = run_phase(
        planner,
        visualizer,
        control,
        "增量重规划：先清除旧代价，再接通备用树枝",
        delay,
        save_only,
    )
    final_path = planner.extract_path()

    if not final_path:
        raise RuntimeError("增加动态障碍后不存在可行路径。")

    visualizer.draw(
        "重规划完成",
        repair_steps,
        expanded=repair_expanded,
        path=final_path,
        message=(
            f"动态障碍：{dynamic_obstacle}\n"
            "新路径改走下方通道。注意：树中绿色节点可以很多，"
            "但只有橙色树边对应最终路径。"
        ),
    )
    visualizer.save(output)

    print(f"初始规划处理节点数: {steps}")
    print(f"初始路径: {initial_path}")
    print(f"动态障碍: {dynamic_obstacle}")
    print(f"增量重规划处理节点数: {repair_steps}")
    print(f"新路径: {final_path}")
    print(f"可视化结果已保存: {output}")

    if not save_only and not control.closed:
        plt.ioff()
        plt.show()
    else:
        plt.close(visualizer.figure)


def parse_args():
    parser = argparse.ArgumentParser(description="树状图可视化 LPA* 算法")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="动画每一步的间隔秒数，默认 0.8",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="不显示窗口，只生成最终 PNG",
    )
    parser.add_argument(
        "--output",
        default="lpa_tree_visualization.png",
        help="结果图片路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_demo(
        delay=max(arguments.delay, 0.0),
        save_only=arguments.save_only,
        output=arguments.output,
    )
