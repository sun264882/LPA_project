import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

    def neighbors(self, node):
        """
        三维 26 邻域。
        """
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

    def world_position(self, node):
        return np.array(node, dtype=float) * self.resolution


# ============================================================
# 2. LPA* 三维路径规划算法
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

        self.expanded_nodes = []
        self.search_snapshots = []

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
        """
        启发函数：欧氏距离。
        """
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
        """
        获取 OPEN 表最小 key，但不弹出有效节点。
        会清理 heap 中已经失效的旧条目。
        """
        while self.open_heap:
            key, node = self.open_heap[0]

            if node in self.open_dict and self.open_dict[node] == key:
                return key

            heapq.heappop(self.open_heap)

        return float("inf"), float("inf")

    def pop_open(self):
        """
        弹出 OPEN 表中当前有效的最小节点。
        """
        while self.open_heap:
            key, node = heapq.heappop(self.open_heap)

            if node in self.open_dict and self.open_dict[node] == key:
                del self.open_dict[node]
                return node

        return None

    def update_vertex(self, node):
        """
        更新节点 rhs 值。
        对于普通静态图，前驱节点与邻居节点相同。
        """
        if node != self.start:
            min_rhs = float("inf")

            for pred, cost in self.map.neighbors(node):
                candidate = self.get_g(pred) + cost

                if candidate < min_rhs:
                    min_rhs = candidate

            self.set_rhs(node, min_rhs)

        self.remove_from_open(node)

        if self.get_g(node) != self.get_rhs(node):
            self.insert(node)

    def compute_shortest_path(self, max_iterations=200000, snapshot_interval=30):
        iteration = 0

        while (
            self.key_less(self.top_key(), self.calculate_key(self.goal))
            or self.get_rhs(self.goal) != self.get_g(self.goal)
        ):
            u = self.pop_open()

            if u is None:
                print("OPEN 表为空，搜索终止。")
                break

            if self.get_g(u) > self.get_rhs(u):
                self.set_g(u, self.get_rhs(u))
                self.expanded_nodes.append(u)

                for succ, _ in self.map.neighbors(u):
                    self.update_vertex(succ)

            else:
                self.set_g(u, float("inf"))
                self.update_vertex(u)

                for succ, _ in self.map.neighbors(u):
                    self.update_vertex(succ)

            iteration += 1

            if iteration % snapshot_interval == 0:
                self.search_snapshots.append(list(self.expanded_nodes))

            if iteration >= max_iterations:
                print("警告：LPA* 达到最大迭代次数，搜索提前终止。")
                break

        self.search_snapshots.append(list(self.expanded_nodes))

    def extract_path(self):
        """
        正确路径提取方式：

        当前 LPA* 从 start 开始传播代价，因此：
            g(node) 表示 start 到 node 的最短代价估计。

        所以路径应当从 goal 反向回溯：
            current = goal
            每次寻找使 g(pred) + cost(pred, current) 最小的前驱节点
            直到回到 start

        最后再 reverse。
        """
        if self.get_g(self.goal) == float("inf"):
            return []

        path = [self.goal]
        current = self.goal

        visited = set()
        visited.add(current)

        max_path_len = self.map.size_x * self.map.size_y * self.map.size_z

        while current != self.start:
            best_pred = None
            best_value = float("inf")

            for pred, cost in self.map.neighbors(current):
                g_pred = self.get_g(pred)

                if g_pred == float("inf"):
                    continue

                candidate = g_pred + cost

                if candidate < best_value:
                    best_value = candidate
                    best_pred = pred

            if best_pred is None:
                print("路径提取失败：无法从当前节点找到有效前驱。")
                return []

            if best_pred in visited:
                print("路径提取失败：出现回溯环路。")
                return []

            path.append(best_pred)
            visited.add(best_pred)
            current = best_pred

            if len(path) > max_path_len:
                print("路径提取失败：路径长度异常。")
                return []

        path.reverse()
        return path


# ============================================================
# 3. 四旋翼三自由度简化动力学模型
# ============================================================

class Quadrotor3DOF:
    """
    三自由度四旋翼简化模型。

    状态：
        p = [x, y, z]
        v = [vx, vy, vz]

    控制输入：
        a = [ax, ay, az]

    动力学：
        p_dot = v
        v_dot = a

    说明：
        该模型用于路径规划与轨迹跟踪演示。
        没有显式建模横滚角、俯仰角、偏航角和电机转速。
    """

    def __init__(self, position):
        self.p = np.array(position, dtype=float)
        self.v = np.zeros(3, dtype=float)

        self.max_acc = 3.0
        self.max_speed = 3.0

    def step(self, target, dt):
        target = np.array(target, dtype=float)

        kp = 3.0
        kd = 2.2

        error_p = target - self.p
        error_v = -self.v

        acc_cmd = kp * error_p + kd * error_v

        acc_norm = np.linalg.norm(acc_cmd)
        if acc_norm > self.max_acc:
            acc_cmd = acc_cmd / acc_norm * self.max_acc

        self.v += acc_cmd * dt

        speed = np.linalg.norm(self.v)
        if speed > self.max_speed:
            self.v = self.v / speed * self.max_speed

        self.p += self.v * dt

        return self.p.copy(), self.v.copy(), acc_cmd.copy()


# ============================================================
# 4. 路径处理
# ============================================================

def grid_path_to_world_path(grid_map, path):
    return [grid_map.world_position(node) for node in path]


def densify_path(path, step=0.15):
    """
    将离散路径加密，方便四旋翼连续跟踪。
    """
    dense = []

    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)

        dist = np.linalg.norm(p1 - p0)
        n = max(2, int(dist / step))

        for j in range(n):
            alpha = j / n
            dense.append((1 - alpha) * p0 + alpha * p1)

    dense.append(np.array(path[-1], dtype=float))
    return dense


# ============================================================
# 5. 可视化函数
# ============================================================

def draw_box(ax, x0, x1, y0, y1, z0, z1, alpha=0.25):
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

    box = Poly3DCollection(faces, alpha=alpha)
    ax.add_collection3d(box)


def setup_3d_axis(ax, grid_map, title):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(0, grid_map.size_x)
    ax.set_ylim(0, grid_map.size_y)
    ax.set_zlim(0, grid_map.size_z)

    ax.view_init(elev=28, azim=-55)
    ax.grid(True)


def plot_obstacles(ax, obstacle_boxes):
    for box in obstacle_boxes:
        x0, x1, y0, y1, z0, z1 = box
        draw_box(ax, x0, x1, y0, y1, z0, z1, alpha=0.3)


def draw_quadrotor(ax, position, size=0.35):
    x, y, z = position

    arm1_x = [x - size, x + size]
    arm1_y = [y, y]
    arm1_z = [z, z]

    arm2_x = [x, x]
    arm2_y = [y - size, y + size]
    arm2_z = [z, z]

    line1, = ax.plot(arm1_x, arm1_y, arm1_z, linewidth=3)
    line2, = ax.plot(arm2_x, arm2_y, arm2_z, linewidth=3)

    rotor_points = np.array([
        [x - size, y, z],
        [x + size, y, z],
        [x, y - size, z],
        [x, y + size, z],
    ])

    rotors = ax.scatter(
        rotor_points[:, 0],
        rotor_points[:, 1],
        rotor_points[:, 2],
        s=40
    )

    body = ax.scatter([x], [y], [z], s=70)

    return [line1, line2, rotors, body]


# ============================================================
# 6. 主程序
# ============================================================

def main():
    # ----------------------------
    # 地图参数
    # ----------------------------
    size_x, size_y, size_z = 24, 18, 10
    grid_map = GridMap3D(size_x, size_y, size_z, resolution=1.0)

    start = (1, 1, 1)
    goal = (22, 15, 8)

    obstacle_boxes = [
        (5, 9, 3, 14, 0, 6),
        (12, 15, 0, 10, 2, 8),
        (16, 20, 8, 12, 0, 7),
        (9, 12, 13, 17, 3, 9),
    ]

    for box in obstacle_boxes:
        x0, x1, y0, y1, z0, z1 = box
        grid_map.set_obstacle_box((x0, x1), (y0, y1), (z0, z1))

    # 确保起点和终点不是障碍物
    grid_map.occupied[start] = False
    grid_map.occupied[goal] = False

    print("start free:", grid_map.is_free(start))
    print("goal free:", grid_map.is_free(goal))

    # ----------------------------
    # LPA* 路径规划
    # ----------------------------
    planner = LPAStar3D(grid_map, start, goal)
    planner.compute_shortest_path(
        max_iterations=200000,
        snapshot_interval=30
    )

    grid_path = planner.extract_path()

    if not grid_path:
        print("未找到可行路径。")
        return

    world_path = grid_path_to_world_path(grid_map, grid_path)
    dense_path = densify_path(world_path, step=0.15)

    print("========== LPA* 规划结果 ==========")
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"路径节点数: {len(grid_path)}")
    print(f"扩展节点数: {len(planner.expanded_nodes)}")
    print(f"路径总代价: {planner.get_g(goal):.3f}")
    print("===================================")

    # ----------------------------
    # 第一部分：LPA* 搜索过程动画
    # ----------------------------
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection="3d")

    path_np = np.array(world_path)
    start_np = np.array(start)
    goal_np = np.array(goal)

    def update_search(frame):
        ax1.clear()
        setup_3d_axis(ax1, grid_map, "LPA* 三维路径规划过程")
        plot_obstacles(ax1, obstacle_boxes)

        expanded = planner.search_snapshots[frame]

        if len(expanded) > 0:
            exp_np = np.array(expanded)
            ax1.scatter(
                exp_np[:, 0],
                exp_np[:, 1],
                exp_np[:, 2],
                s=8,
                alpha=0.35,
                label="Expanded nodes"
            )

        ax1.scatter(
            [start_np[0]], [start_np[1]], [start_np[2]],
            s=90,
            marker="o",
            label="Start"
        )

        ax1.scatter(
            [goal_np[0]], [goal_np[1]], [goal_np[2]],
            s=120,
            marker="*",
            label="Goal"
        )

        if frame == len(planner.search_snapshots) - 1:
            ax1.plot(
                path_np[:, 0],
                path_np[:, 1],
                path_np[:, 2],
                linewidth=3,
                label="LPA* path"
            )

        ax1.legend(loc="upper left")

    ani1 = FuncAnimation(
        fig1,
        update_search,
        frames=len(planner.search_snapshots),
        interval=80,
        repeat=False
    )

    # 防止动画对象被垃圾回收
    keep_animation_1 = ani1

    # ----------------------------
    # 第二部分：四旋翼沿路径飞行动画
    # ----------------------------
    quad = Quadrotor3DOF(position=world_path[0])

    dt = 0.04
    target_index = 0
    quad_positions = []

    max_steps = 4000

    for _ in range(max_steps):
        target = dense_path[target_index]
        pos, vel, acc = quad.step(target, dt)
        quad_positions.append(pos)

        if np.linalg.norm(pos - target) < 0.18:
            target_index += 1

            if target_index >= len(dense_path):
                break

    quad_positions = np.array(quad_positions)

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection="3d")

    def update_quad(frame):
        ax2.clear()
        setup_3d_axis(ax2, grid_map, "四旋翼三自由度模型沿 LPA* 路径飞行")
        plot_obstacles(ax2, obstacle_boxes)

        ax2.scatter(
            [start_np[0]], [start_np[1]], [start_np[2]],
            s=90,
            marker="o",
            label="Start"
        )

        ax2.scatter(
            [goal_np[0]], [goal_np[1]], [goal_np[2]],
            s=120,
            marker="*",
            label="Goal"
        )

        ax2.plot(
            path_np[:, 0],
            path_np[:, 1],
            path_np[:, 2],
            linewidth=2,
            linestyle="--",
            label="LPA* path"
        )

        if frame > 1:
            traj = quad_positions[:frame]
            ax2.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                linewidth=2,
                label="Quadrotor trajectory"
            )

        draw_quadrotor(ax2, quad_positions[frame], size=0.45)

        ax2.legend(loc="upper left")

    ani2 = FuncAnimation(
        fig2,
        update_quad,
        frames=len(quad_positions),
        interval=25,
        repeat=False
    )

    # 防止动画对象被垃圾回收
    keep_animation_2 = ani2

    plt.show()


if __name__ == "__main__":
    main()