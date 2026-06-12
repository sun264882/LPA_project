import heapq
import math
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# 0. Matplotlib 中文字体设置
# ============================================================

matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 三维栅格地图
# ============================================================

class GridMap3D:
    def __init__(self, size_x, size_y, size_z, resolution=1.0):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.resolution = resolution

        # True 表示障碍物
        self.occupied = np.zeros((size_x, size_y, size_z), dtype=bool)

    def in_bounds(self, node):
        x, y, z = node
        return (
            0 <= x < self.size_x and
            0 <= y < self.size_y and
            0 <= z < self.size_z
        )

    def is_free(self, node):
        return self.in_bounds(node) and not self.occupied[node]

    def set_obstacle_box(self, x_range, y_range, z_range):
        x0, x1 = x_range
        y0, y1 = y_range
        z0, z1 = z_range

        self.occupied[x0:x1, y0:y1, z0:z1] = True

    def add_obstacle_cell(self, node):
        if self.in_bounds(node):
            self.occupied[node] = True

    def neighbors(self, node):
        """
        三维 26 邻域。
        """
        if not self.is_free(node):
            return []

        x, y, z = node
        result = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nxt = (x + dx, y + dy, z + dz)

                    if self.is_free(nxt):
                        cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                        result.append((nxt, cost))

        return result

    def local_cells(self, node, radius=1):
        """
        返回 node 周围局部区域，用于 LPA* 障碍物变化后的局部更新。
        """
        x, y, z = node
        cells = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nxt = (x + dx, y + dy, z + dz)
                    if self.in_bounds(nxt):
                        cells.append(nxt)

        return cells

    def world_position(self, node):
        return np.array(node, dtype=float) * self.resolution


# ============================================================
# 2. A* 三维路径规划
# ============================================================

class AStar3D:
    def __init__(self, grid_map):
        self.map = grid_map

    @staticmethod
    def heuristic(node, goal):
        return math.dist(node, goal)

    def plan(self, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(start, goal), 0.0, start))

        g = {start: 0.0}
        parent = {}

        closed = set()
        expanded_count = 0

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            closed.add(current)
            expanded_count += 1

            if current == goal:
                break

            for neighbor, cost in self.map.neighbors(current):
                new_g = g[current] + cost

                if new_g < g.get(neighbor, float("inf")):
                    g[neighbor] = new_g
                    parent[neighbor] = current

                    f = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, new_g, neighbor))

        if goal not in g:
            return [], expanded_count, float("inf")

        path = [goal]
        current = goal

        while current != start:
            current = parent[current]
            path.append(current)

        path.reverse()

        return path, expanded_count, g[goal]


# ============================================================
# 3. LPA* 三维路径规划
# ============================================================

class LPAStar3D:
    def __init__(self, grid_map, start, goal):
        self.map = grid_map
        self.start = start
        self.goal = goal

        self.g = {}
        self.rhs = {}

        self.open_heap = []
        self.open_dict = {}

        self.total_expanded_count = 0

        self.g[self.start] = float("inf")
        self.rhs[self.start] = 0.0

        self.insert(self.start)

    def get_g(self, node):
        return self.g.get(node, float("inf"))

    def get_rhs(self, node):
        return self.rhs.get(node, float("inf"))

    def set_g(self, node, value):
        self.g[node] = value

    def set_rhs(self, node, value):
        self.rhs[node] = value

    def heuristic(self, node):
        return math.dist(node, self.goal)

    def calculate_key(self, node):
        value = min(self.get_g(node), self.get_rhs(node))
        return value + self.heuristic(node), value

    @staticmethod
    def key_less(k1, k2):
        if k1[0] < k2[0]:
            return True
        if k1[0] == k2[0] and k1[1] < k2[1]:
            return True
        return False

    def insert(self, node):
        key = self.calculate_key(node)
        self.open_dict[node] = key
        heapq.heappush(self.open_heap, (key, node))

    def remove_from_open(self, node):
        if node in self.open_dict:
            del self.open_dict[node]

    def top_key(self):
        while self.open_heap:
            key, node = self.open_heap[0]

            if node in self.open_dict and self.open_dict[node] == key:
                return key

            heapq.heappop(self.open_heap)

        return float("inf"), float("inf")

    def pop_open(self):
        while self.open_heap:
            key, node = heapq.heappop(self.open_heap)

            if node in self.open_dict and self.open_dict[node] == key:
                del self.open_dict[node]
                return node

        return None

    def update_vertex(self, node):
        if not self.map.in_bounds(node):
            return

        # 如果节点已经变成障碍物，直接失效
        if not self.map.is_free(node):
            self.set_g(node, float("inf"))
            self.set_rhs(node, float("inf"))
            self.remove_from_open(node)
            return

        if node != self.start:
            min_rhs = float("inf")

            for pred, cost in self.map.neighbors(node):
                candidate = self.get_g(pred) + cost

                if candidate < min_rhs:
                    min_rhs = candidate

            self.set_rhs(node, min_rhs)

        self.remove_from_open(node)

        if abs(self.get_g(node) - self.get_rhs(node)) > 1e-9:
            self.insert(node)

    def compute_shortest_path(self, max_iterations=1000000):
        expanded_this_run = 0
        iteration = 0

        while (
            self.key_less(self.top_key(), self.calculate_key(self.goal))
            or abs(self.get_rhs(self.goal) - self.get_g(self.goal)) > 1e-9
        ):
            current = self.pop_open()

            if current is None:
                break

            if self.get_g(current) > self.get_rhs(current):
                self.set_g(current, self.get_rhs(current))

                expanded_this_run += 1
                self.total_expanded_count += 1

                for succ, _ in self.map.neighbors(current):
                    self.update_vertex(succ)

            else:
                self.set_g(current, float("inf"))
                self.update_vertex(current)

                for succ, _ in self.map.neighbors(current):
                    self.update_vertex(succ)

            iteration += 1

            if iteration >= max_iterations:
                print("警告：LPA* 达到最大迭代次数。")
                break

        return expanded_this_run

    def notify_obstacle_changes(self, changed_cells):
        """
        地图中某些栅格变为障碍物后，LPA* 只更新这些栅格及其邻域。
        这一步是 LPA* 相比 A* 的核心优势。
        """
        affected = set()

        for cell in changed_cells:
            if not self.map.in_bounds(cell):
                continue

            self.set_g(cell, float("inf"))
            self.set_rhs(cell, float("inf"))
            self.remove_from_open(cell)

            affected.add(cell)

            for local_cell in self.map.local_cells(cell, radius=1):
                affected.add(local_cell)

        for node in affected:
            self.update_vertex(node)

    def extract_path(self):
        if self.get_g(self.goal) == float("inf"):
            return []

        path = [self.goal]
        current = self.goal

        visited = {current}
        max_path_len = self.map.size_x * self.map.size_y * self.map.size_z

        while current != self.start:
            best_pred = None
            best_value = float("inf")

            for pred, cost in self.map.neighbors(current):
                candidate = self.get_g(pred) + cost

                if candidate < best_value:
                    best_value = candidate
                    best_pred = pred

            if best_pred is None:
                return []

            if best_pred in visited:
                return []

            path.append(best_pred)
            visited.add(best_pred)
            current = best_pred

            if len(path) > max_path_len:
                return []

        path.reverse()
        return path


# ============================================================
# 4. 地图与动态障碍物设置
# ============================================================

def create_demo_map():
    grid_map = GridMap3D(40, 30, 12, resolution=1.0)

    # 固定障碍物：三组墙体，每组墙体留出若干通道
    grid_map.set_obstacle_box((8, 10), (0, 22), (0, 9))
    grid_map.occupied[8:10, 4:8, 1:6] = False
    grid_map.occupied[8:10, 18:22, 5:10] = False

    grid_map.set_obstacle_box((18, 20), (8, 30), (2, 12))
    grid_map.occupied[18:20, 12:16, 3:8] = False
    grid_map.occupied[18:20, 23:27, 0:5] = False

    grid_map.set_obstacle_box((28, 30), (0, 24), (0, 8))
    grid_map.occupied[28:30, 3:7, 2:7] = False
    grid_map.occupied[28:30, 15:20, 4:10] = False

    return grid_map


def add_dynamic_obstacle_around_path(grid_map, path, ratio, start, goal):
    """
    在当前路径上的某个位置附近添加局部障碍物。
    ratio 表示选择路径上的比例位置。
    """
    if len(path) < 3:
        return []

    index = int(len(path) * ratio)
    index = max(1, min(index, len(path) - 2))

    cx, cy, cz = path[index]

    changed_cells = []

    # 添加一个小块三维障碍物
    for x in range(cx - 1, cx + 2):
        for y in range(cy - 2, cy + 3):
            for z in range(cz - 1, cz + 2):
                cell = (x, y, z)

                if not grid_map.in_bounds(cell):
                    continue

                if cell == start or cell == goal:
                    continue

                if not grid_map.occupied[cell]:
                    grid_map.add_obstacle_cell(cell)
                    changed_cells.append(cell)

    return changed_cells


# ============================================================
# 5. 可视化函数
# ============================================================

def draw_box(ax, x0, x1, y0, y1, z0, z1, color="gray", alpha=0.25):
    vertices = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [0, 3, 7, 4]],
    ]

    box = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor="k", linewidths=0.2)
    ax.add_collection3d(box)


def setup_axis_3d(ax, grid_map, title):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(0, grid_map.size_x)
    ax.set_ylim(0, grid_map.size_y)
    ax.set_zlim(0, grid_map.size_z)

    ax.view_init(elev=28, azim=-55)
    ax.grid(True)


def plot_static_obstacles(ax):
    """
    画出固定障碍物。
    """
    draw_box(ax, 8, 10, 0, 22, 0, 9, color="gray", alpha=0.20)
    draw_box(ax, 18, 20, 8, 30, 2, 12, color="gray", alpha=0.20)
    draw_box(ax, 28, 30, 0, 24, 0, 8, color="gray", alpha=0.20)


def plot_dynamic_obstacles(ax, dynamic_cells):
    """
    动态障碍物用散点显示。
    """
    if len(dynamic_cells) == 0:
        return

    arr = np.array(dynamic_cells)

    ax.scatter(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        s=20,
        c="red",
        marker="s",
        alpha=0.65,
        label="Dynamic obstacles"
    )


def plot_path(ax, path, label, color, linewidth=3, linestyle="-"):
    if not path:
        return

    arr = np.array(path)

    ax.plot(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label
    )


def plot_comparison_charts(records):
    labels = [record["stage"] for record in records]

    astar_expanded = [record["astar_expanded"] for record in records]
    lpastar_expanded = [record["lpastar_expanded"] for record in records]

    astar_time = [record["astar_time_ms"] for record in records]
    lpastar_time = [record["lpastar_time_ms"] for record in records]

    x = np.arange(len(labels))
    width = 0.35

    # ----------------------------
    # 扩展节点数对比
    # ----------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 5))

    ax1.bar(x - width / 2, astar_expanded, width, label="A* 从零搜索")
    ax1.bar(x + width / 2, lpastar_expanded, width, label="LPA* 增量修复")

    ax1.set_title("A* 与 LPA* 每次规划扩展节点数对比")
    ax1.set_xlabel("规划阶段")
    ax1.set_ylabel("扩展节点数")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # ----------------------------
    # 搜索耗时对比
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 5))

    ax2.bar(x - width / 2, astar_time, width, label="A* 从零搜索")
    ax2.bar(x + width / 2, lpastar_time, width, label="LPA* 增量修复")

    ax2.set_title("A* 与 LPA* 每次规划耗时对比")
    ax2.set_xlabel("规划阶段")
    ax2.set_ylabel("耗时 / ms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # ----------------------------
    # 累计扩展节点数对比
    # ----------------------------
    fig3, ax3 = plt.subplots(figsize=(11, 5))

    astar_cumsum = np.cumsum(astar_expanded)
    lpastar_cumsum = np.cumsum(lpastar_expanded)

    ax3.plot(labels, astar_cumsum, marker="o", label="A* 累计扩展节点数")
    ax3.plot(labels, lpastar_cumsum, marker="s", label="LPA* 累计扩展节点数")

    ax3.set_title("A* 与 LPA* 累计搜索代价对比")
    ax3.set_xlabel("规划阶段")
    ax3.set_ylabel("累计扩展节点数")
    ax3.legend()
    ax3.grid(alpha=0.3)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    start = (1, 2, 1)
    goal = (38, 27, 9)

    grid_map = create_demo_map()

    grid_map.occupied[start] = False
    grid_map.occupied[goal] = False

    astar = AStar3D(grid_map)
    lpastar = LPAStar3D(grid_map, start, goal)

    records = []
    dynamic_obstacles = []

    # ========================================================
    # 第 1 阶段：初始规划
    # ========================================================

    t0 = time.perf_counter()
    astar_path, astar_expanded, astar_cost = astar.plan(start, goal)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    lpastar_expanded = lpastar.compute_shortest_path()
    lpastar_path = lpastar.extract_path()
    lpastar_cost = lpastar.get_g(goal)
    t3 = time.perf_counter()

    if not astar_path or not lpastar_path:
        print("初始地图中未找到可行路径。")
        return

    records.append({
        "stage": "初始规划",
        "astar_expanded": astar_expanded,
        "lpastar_expanded": lpastar_expanded,
        "astar_time_ms": (t1 - t0) * 1000,
        "lpastar_time_ms": (t3 - t2) * 1000,
        "astar_cost": astar_cost,
        "lpastar_cost": lpastar_cost
    })

    current_lpa_path = lpastar_path

    # ========================================================
    # 第 2 阶段：动态加入障碍物并重规划
    # ========================================================

    obstacle_ratios = [0.25, 0.45, 0.65, 0.80]

    for i, ratio in enumerate(obstacle_ratios, start=1):
        changed_cells = add_dynamic_obstacle_around_path(
            grid_map,
            current_lpa_path,
            ratio,
            start,
            goal
        )

        dynamic_obstacles.extend(changed_cells)

        # LPA* 只通知局部变化
        lpastar.notify_obstacle_changes(changed_cells)

        # A* 从零重新搜索
        t0 = time.perf_counter()
        astar_path, astar_expanded, astar_cost = astar.plan(start, goal)
        t1 = time.perf_counter()

        # LPA* 增量修复
        t2 = time.perf_counter()
        lpastar_expanded = lpastar.compute_shortest_path()
        lpastar_path = lpastar.extract_path()
        lpastar_cost = lpastar.get_g(goal)
        t3 = time.perf_counter()

        if not astar_path or not lpastar_path:
            print(f"第 {i} 次动态重规划后未找到可行路径。")
            break

        records.append({
            "stage": f"重规划 {i}",
            "astar_expanded": astar_expanded,
            "lpastar_expanded": lpastar_expanded,
            "astar_time_ms": (t1 - t0) * 1000,
            "lpastar_time_ms": (t3 - t2) * 1000,
            "astar_cost": astar_cost,
            "lpastar_cost": lpastar_cost
        })

        current_lpa_path = lpastar_path

    # ========================================================
    # 打印对比结果
    # ========================================================

    print("\n================ A* 与 LPA* 对比结果 ================")
    print(f"{'阶段':<10} | {'A*扩展':>8} | {'LPA*扩展':>10} | {'A*耗时/ms':>12} | {'LPA*耗时/ms':>14} | {'A*代价':>10} | {'LPA*代价':>10}")
    print("-" * 95)

    for record in records:
        print(
            f"{record['stage']:<10} | "
            f"{record['astar_expanded']:>8} | "
            f"{record['lpastar_expanded']:>10} | "
            f"{record['astar_time_ms']:>12.3f} | "
            f"{record['lpastar_time_ms']:>14.3f} | "
            f"{record['astar_cost']:>10.3f} | "
            f"{record['lpastar_cost']:>10.3f}"
        )

    astar_total_expanded = sum(record["astar_expanded"] for record in records)
    lpastar_total_expanded = sum(record["lpastar_expanded"] for record in records)

    astar_total_time = sum(record["astar_time_ms"] for record in records)
    lpastar_total_time = sum(record["lpastar_time_ms"] for record in records)

    print("-" * 95)
    print(f"A* 累计扩展节点数:   {astar_total_expanded}")
    print(f"LPA* 累计扩展节点数: {lpastar_total_expanded}")
    print(f"A* 累计耗时/ms:      {astar_total_time:.3f}")
    print(f"LPA* 累计耗时/ms:    {lpastar_total_time:.3f}")

    if lpastar_total_expanded > 0:
        ratio = astar_total_expanded / lpastar_total_expanded
        print(f"扩展节点数优势倍数:  A* / LPA* = {ratio:.2f}")

    print("=====================================================\n")

    # ========================================================
    # 三维路径结果可视化
    # ========================================================

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    setup_axis_3d(ax, grid_map, "A* 与 LPA* 三维动态重规划结果对比")

    plot_static_obstacles(ax)
    plot_dynamic_obstacles(ax, dynamic_obstacles)

    final_astar_path = astar_path
    final_lpastar_path = lpastar_path

    plot_path(
        ax,
        final_astar_path,
        label="A* final path",
        color="blue",
        linewidth=2,
        linestyle="--"
    )

    plot_path(
        ax,
        final_lpastar_path,
        label="LPA* final path",
        color="green",
        linewidth=3,
        linestyle="-"
    )

    ax.scatter(
        [start[0]],
        [start[1]],
        [start[2]],
        c="orange",
        s=100,
        marker="o",
        label="Start"
    )

    ax.scatter(
        [goal[0]],
        [goal[1]],
        [goal[2]],
        c="purple",
        s=130,
        marker="*",
        label="Goal"
    )

    ax.legend(loc="upper left")

    # ========================================================
    # 指标图
    # ========================================================

    plot_comparison_charts(records)

    plt.show()


if __name__ == "__main__":
    main()