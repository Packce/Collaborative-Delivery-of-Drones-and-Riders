"""无人机路径代价模型。

该模块包含：
1. 地形插值模型（Terrain）
2. 建筑物障碍物模型（Obstacle）
3. 路径总代价计算器（UAVPathCostCalculator）
4. 从 CSV 自动加载地形与建筑数据的工具函数
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, pi, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Obstacle:
    """圆柱体障碍物模型（建筑物简化）。

    属性:
        x: 障碍物中心点 X 坐标（投影坐标，单位通常为米）
        y: 障碍物中心点 Y 坐标（投影坐标，单位通常为米）
        z: 建筑物相对地面高度（米）
        r: 建筑物等效半径（米）
    """

    x: float
    y: float
    z: float
    r: float


class Terrain:
    """地形高程模型，支持规则网格上的双线性插值。"""

    def __init__(
        self,
        elevation_matrix: np.ndarray,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        x_coords: Optional[Sequence[float]] = None,
        y_coords: Optional[Sequence[float]] = None,
    ):
        """初始化地形对象。

        参数:
            elevation_matrix: 地形高程矩阵，形状为 (ny, nx)
            x_range: 当不提供 x_coords 时使用，表示 X 方向坐标范围 (min_x, max_x)
            y_range: 当不提供 y_coords 时使用，表示 Y 方向坐标范围 (min_y, max_y)
            x_coords: 每一列对应的实际 X 坐标数组（长度需等于 nx）
            y_coords: 每一行对应的实际 Y 坐标数组（长度需等于 ny）

        说明:
            - 优先使用 x_coords/y_coords
            - 若使用 x_range/y_range，则按线性均匀网格自动生成坐标。
        """

        self.elevation = np.asarray(elevation_matrix, dtype=float)
        if self.elevation.ndim != 2:
            raise ValueError("elevation_matrix must be 2D")

        self.ny, self.nx = self.elevation.shape
        if self.nx < 2 or self.ny < 2:
            raise ValueError("elevation_matrix must be at least 2x2 for interpolation")

        if x_coords is not None and y_coords is not None:
            x_arr = np.asarray(x_coords, dtype=float)
            y_arr = np.asarray(y_coords, dtype=float)
            if x_arr.size != self.nx:
                raise ValueError(f"x_coords length {x_arr.size} != nx {self.nx}")
            if y_arr.size != self.ny:
                raise ValueError(f"y_coords length {y_arr.size} != ny {self.ny}")
            if np.any(np.diff(x_arr) <= 0):
                raise ValueError("x_coords must be strictly increasing")
            if np.any(np.diff(y_arr) <= 0):
                raise ValueError("y_coords must be strictly increasing")
            self.x_coords = x_arr
            self.y_coords = y_arr
        elif x_range is not None and y_range is not None:
            x_min, x_max = x_range
            y_min, y_max = y_range
            if x_max <= x_min or y_max <= y_min:
                raise ValueError("x_range/y_range must be (min, max) with max > min")
            self.x_coords = np.linspace(float(x_min), float(x_max), self.nx)
            self.y_coords = np.linspace(float(y_min), float(y_max), self.ny)
        else:
            raise ValueError("Provide either (x_coords, y_coords) or (x_range, y_range)")

        self.x_min = float(self.x_coords[0])
        self.x_max = float(self.x_coords[-1])
        self.y_min = float(self.y_coords[0])
        self.y_max = float(self.y_coords[-1])

    def get_elevation(self, x: float, y: float) -> float:
        """查询任意坐标点 (x, y) 的地形高程。

        参数:
            x: 查询点 X 坐标
            y: 查询点 Y 坐标

        返回:
            双线性插值后的地形高程值。

        说明:
            - 超出边界会先裁剪到地形范围内。
            - 使用网格坐标与实际坐标映射后进行双线性插值。
        """

        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.y_min, self.y_max))

        xi = float(np.interp(x, self.x_coords, np.arange(self.nx, dtype=float)))
        yi = float(np.interp(y, self.y_coords, np.arange(self.ny, dtype=float)))

        ix = int(np.floor(xi))
        iy = int(np.floor(yi))

        ix = min(max(ix, 0), self.nx - 2)
        iy = min(max(iy, 0), self.ny - 2)

        dx = xi - ix
        dy = yi - iy

        h00 = self.elevation[iy, ix]
        h10 = self.elevation[iy, ix + 1]
        h01 = self.elevation[iy + 1, ix]
        h11 = self.elevation[iy + 1, ix + 1]

        h0 = h00 * (1.0 - dx) + h10 * dx
        h1 = h01 * (1.0 - dx) + h11 * dx
        return float(h0 * (1.0 - dy) + h1 * dy)


class UAVPathCostCalculator:
    """无人机路径代价计算器。"""

    def __init__(
        self,
        terrain: Terrain,
        obstacles: List[Obstacle],
        c_T: float = 1000.0,
        c_B: float = 100.0,
        c_hr: float = 20.0,
        c_theta: float = 20.0 / pi,
        k_nearest: int = 4,
        ignore_first_collision: bool = True,
        ignore_last_collision: bool = True,
        customers: Optional[List[Dict[str, Any]]] = None,
        candidate_points: Optional[List[Tuple[float, float]]] = None,
        merchants: Optional[List[Dict[str, Any]]] = None,
        c_delivery: float = 1.0,
        rider_speed: float = 15.0,
        drone_speed: float = 50.0,
    ):
        """初始化路径代价计算器。

        参数:
            terrain: 地形对象
            obstacles: 障碍物列表
            c_T: 地形约束惩罚系数
            c_B: 障碍物碰撞惩罚系数
            c_hr: 飞行高度奖励系数（高于建筑顶部时给负代价奖励）
            c_theta: 转角约束惩罚系数
            k_nearest: 每个路径点仅评估最近的 k 个障碍物，用于提速
            ignore_first_collision: 是否忽略全路径的第一次碰撞惩罚
            ignore_last_collision: 是否忽略全路径的最后一次碰撞惩罚
            customers: 顾客数据列表，每项包含 x, y 坐标
            candidate_points: 候选点坐标列表，用于计算配送时间
            merchants: 商家数据列表，每项包含 x, y 坐标
            c_delivery: 配送时间代价权重系数
            rider_speed: 骑手平均速度（km/h），默认15km/h
            drone_speed: 无人机平均速度（km/h），默认50km/h
        """

        self.terrain = terrain
        self.obstacles = obstacles
        self.c_T = c_T
        self.c_B = c_B
        self.c_hr = c_hr
        self.c_theta = c_theta
        self.k_nearest = max(1, int(k_nearest))
        self.ignore_first_collision = bool(ignore_first_collision)
        self.ignore_last_collision = bool(ignore_last_collision)
        self.customers = customers if customers is not None else []
        self.candidate_points = candidate_points if candidate_points is not None else []
        self.merchants = merchants if merchants is not None else []
        self.c_delivery = c_delivery
        self.rider_speed = rider_speed
        self.drone_speed = drone_speed

        if obstacles:
            self._obs_x = np.array([obs.x for obs in obstacles], dtype=float)
            self._obs_y = np.array([obs.y for obs in obstacles], dtype=float)
            self._obs_r = np.array([obs.r for obs in obstacles], dtype=float)
            self._obs_top = np.array(
                [self.terrain.get_elevation(obs.x, obs.y) + obs.z for obs in obstacles],
                dtype=float,
            )
        else:
            self._obs_x = np.empty(0, dtype=float)
            self._obs_y = np.empty(0, dtype=float)
            self._obs_r = np.empty(0, dtype=float)
            self._obs_top = np.empty(0, dtype=float)

    @staticmethod
    def _extract_service_points_from_paths(
        paths: List[List[List[Tuple[float, float, float]]]],
    ) -> List[Tuple[float, float]]:
        """从 paths 结构提取实际被服务的候选点坐标集合。"""

        service_points: List[Tuple[float, float]] = []
        for merchant_paths in paths:
            for path in merchant_paths:
                if path:
                    service_points.append((float(path[-1][0]), float(path[-1][1])))
        return service_points

    def _nearest_obstacle_indices(self, point: Tuple[float, float]) -> np.ndarray:
        """返回距离给定点最近的障碍物索引集合。

        参数:
            point: 当前参考点 (x, y)

        返回:
            按距离从近到远排序的障碍物索引数组，长度不超过 k_nearest。
        """

        if self._obs_x.size == 0:
            return np.empty(0, dtype=int)

        px, py = point
        dist2 = (self._obs_x - px) ** 2 + (self._obs_y - py) ** 2
        k = min(self.k_nearest, dist2.size)
        nearest = np.argpartition(dist2, k - 1)[:k]
        nearest = nearest[np.argsort(dist2[nearest])]
        return nearest

    @staticmethod
    def _segment_intersect_cylinder(
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        cx: float,
        cy: float,
        r: float,
    ) -> bool:
        """判断线段与圆柱体水平投影是否相交。

        参数:
            p1: 线段起点 (x, y, H)
            p2: 线段终点 (x, y, H)
            cx: 圆心 X 坐标
            cy: 圆心 Y 坐标
            r: 半径

        返回:
            若线段与圆（障碍物水平投影）相交或接触，返回 True；否则 False。
        """

        x1, y1, _ = p1
        x2, y2, _ = p2

        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx * dx + dy * dy
        if a < 1e-12:
            return (fx * fx + fy * fy) <= r * r

        t = -(fx * dx + fy * dy) / a
        t = max(0.0, min(1.0, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
        return dist2 <= r * r

    def terrain_cost(self, path: List[Tuple[float, float, float]]) -> float:
        """计算地形约束代价。

        规则:
            - 若 H > 120，超高部分按 c_T 惩罚。
            - 若 DH(x,y) < H <= 120，代价为 0。
            - 若 H <= DH(x,y)，低于地形部分按 c_T 惩罚。

        参数:
            path: 路径点列表，每个点为 (x, y, H)

        返回:
            地形约束总成本。
        """

        total = 0.0
        for x, y, H in path:
            dh = self.terrain.get_elevation(x, y)
            if H > 120.0:
                cost = (H - 120.0) * self.c_T
            elif dh < H <= 120.0:
                cost = 0.0
            else:
                cost = (dh - H) * self.c_T
            total += cost
        return float(total)

    def obstacle_collision_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算障碍物相关代价（碰撞惩罚 + 高度奖励）。

        参数:
            path: 中间路径点列表 (x, y, H)
            start: 起点 (x, y, H)
            end: 终点 (x, y, H)

        返回:
            障碍物总代价。

        说明:
            - 对每个航段，仅检查最近 k 个障碍物。
            - 若航段与障碍物相交，则叠加碰撞惩罚 c_B。
            - 若当前点高度高于建筑顶部，可获得 -c_hr 奖励。
            - 若 ignore_first_collision=True，则全路径第一次碰撞惩罚会被忽略。
            - 若 ignore_last_collision=True，则全路径最后一次碰撞惩罚会被忽略。
        """

        full_path = [start] + path + [end]
        collision_flags: List[bool] = []
        height_rewards: List[float] = []

        for i in range(1, len(full_path)):
            p_prev = full_path[i - 1]
            p_curr = full_path[i]

            nearest_idx = self._nearest_obstacle_indices((p_curr[0], p_curr[1]))
            for obs_idx in nearest_idx:
                obs = self.obstacles[int(obs_idx)]

                collide = False
                if (p_curr[0] - obs.x) ** 2 + (p_curr[1] - obs.y) ** 2 <= obs.r ** 2:
                    collide = True
                if (p_prev[0] - obs.x) ** 2 + (p_prev[1] - obs.y) ** 2 <= obs.r ** 2:
                    collide = True
                if self._segment_intersect_cylinder(p_prev, p_curr, obs.x, obs.y, obs.r):
                    collide = True

                H_i = p_curr[2]
                building_top = float(self._obs_top[int(obs_idx)])
                hr = -self.c_hr if H_i >= building_top else 0.0

                collision_flags.append(collide)
                height_rewards.append(hr)

        collision_indices = [idx for idx, is_collision in enumerate(collision_flags) if is_collision]
        ignored_collision_indices = set()

        if self.ignore_first_collision and collision_indices:
            ignored_collision_indices.add(collision_indices[0])
        if self.ignore_last_collision and collision_indices:
            ignored_collision_indices.add(collision_indices[-1])

        total_B = 0.0
        for idx, collide in enumerate(collision_flags):
            if collide and idx not in ignored_collision_indices:
                total_B += self.c_B
            total_B += height_rewards[idx]

        return float(total_B)

    def flight_distance_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算飞行总路程代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            起点 -> 路径点 -> 终点 的三维欧式距离总和。
        """

        if not path:
            return float(
                sqrt(
                    (end[0] - start[0]) ** 2
                    + (end[1] - start[1]) ** 2
                    + (end[2] - start[2]) ** 2
                )
            )

        dist = sqrt(
            (path[0][0] - start[0]) ** 2
            + (path[0][1] - start[1]) ** 2
            + (path[0][2] - start[2]) ** 2
        )

        for i in range(1, len(path)):
            dist += sqrt(
                (path[i][0] - path[i - 1][0]) ** 2
                + (path[i][1] - path[i - 1][1]) ** 2
                + (path[i][2] - path[i - 1][2]) ** 2
            )

        dist += sqrt(
            (end[0] - path[-1][0]) ** 2
            + (end[1] - path[-1][1]) ** 2
            + (end[2] - path[-1][2]) ** 2
        )
        return float(dist)

    def altitude_variation_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算高度变化代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            相邻航段高度差绝对值之和。
        """

        if not path:
            return float(abs(end[2] - start[2]))

        delta = abs(path[0][2] - start[2])
        for i in range(1, len(path)):
            delta += abs(path[i][2] - path[i - 1][2])
        delta += abs(end[2] - path[-1][2])

        return float(delta)

    def turning_angle_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算转角约束代价。

        参数:
            path: 中间路径点列表
            start: 起点
            end: 终点

        返回:
            相邻航段航向角变化绝对值之和，再乘以 c_theta。
        """

        points = [start] + path + [end]
        if len(points) < 3:
            return 0.0

        angles = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            angles.append(atan2(dy, dx))

        total_angle_change = 0.0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            diff = abs((diff + pi) % (2 * pi) - pi)
            total_angle_change += diff

        return float(total_angle_change * self.c_theta)

    def rider_time_cost(
        self,
        paths: List[List[List[Tuple[float, float, float]]]],
    ) -> float:
        """计算从候选点到顾客的配送时间代价。

        参数:
            paths: 路径列表，paths[i][j] 对应 merchants[i] 到候选点 j 的路径

        返回:
            所有顾客到其最近候选点的配送时间之和（小时），乘以权重系数 c_delivery。

        说明:
            - 只考虑 x, y 坐标的距离
            - 配送时间 = 距离 / 骑手平均速度（rider_speed，单位km/h）
            - 距离单位是米，需要除以1000转换为公里
            - 每个候选点可由多个商家共同服务
        """
        if not self.customers:
            return 0.0

        service_points = self._extract_service_points_from_paths(paths)
        if not service_points:
            service_points = [(float(p[0]), float(p[1])) for p in self.candidate_points]
        if not service_points:
            return 0.0

        total_delivery_time = 0.0
        speed_km_per_h = self.rider_speed

        for customer in self.customers:
            cx, cy = customer.get('x', 0.0), customer.get('y', 0.0)

            min_distance = float('inf')
            for px, py in service_points:
                dist = sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < min_distance:
                    min_distance = dist

            delivery_time_hours = (min_distance / 1000.0) / speed_km_per_h
            total_delivery_time += delivery_time_hours

        return float(total_delivery_time * self.c_delivery)

    def drone_time_cost(
        self,
        path: List[Tuple[float, float, float]],
        merchant: Dict[str, Any],
    ) -> float:
        """计算单个商家到候选点的无人机构建时间（未乘权重）。

        参数:
            path: 该商家到候选点的中间路径点列表
            merchant: 商家信息字典，包含 x, y 坐标

        返回:
            该商家无人机到候选点的飞行时间（小时）。

        说明:
            - 距离计算包含 x, y, z 三个坐标
            - 使用无人机速度 drone_speed（km/h）
            - 距离单位是米，需要除以1000转换为公里
        """
        if not path:
            return 0.0

        mx, my = merchant.get('x', 0.0), merchant.get('y', 0.0)
        mz = self.terrain.get_elevation(mx, my)

        first_point = path[0]
        px, py, pz = first_point[0], first_point[1], first_point[2]
        dist_3d = sqrt((mx - px) ** 2 + (my - py) ** 2 + (mz - pz) ** 2)

        for i in range(1, len(path)):
            dist_3d += sqrt(
                (path[i][0] - path[i - 1][0]) ** 2
                + (path[i][1] - path[i - 1][1]) ** 2
                + (path[i][2] - path[i - 1][2]) ** 2
            )

        drone_time_hours = (dist_3d / 1000.0) / self.drone_speed

        return float(drone_time_hours)

    def total_drone_time(
        self,
        paths: List[List[List[Tuple[float, float, float]]]],
    ) -> float:
        """计算所有商家到所有候选点的无人机构建时间总和。

        参数:
            paths: 路径列表，paths[i][j] 对应 merchants[i] 到候选点 j 的路径

        返回:
            所有商家到所有候选点无人机飞行时间之和（小时）。
        """
        if not self.merchants or not paths:
            return 0.0

        total_time = 0.0
        for i, merchant_paths in enumerate(paths):
            if i >= len(self.merchants):
                break
            for path in merchant_paths:
                total_time += self.drone_time_cost(path, self.merchants[i])
        return float(total_time)

    def total_time(
        self,
        paths: List[List[List[Tuple[float, float, float]]]],
    ) -> float:
        """计算总时间（所有商家无人机运送时间 + 骑手配送时间）。

        参数:
            paths: 路径列表，paths[i][j] 对应 merchants[i] 到候选点 j 的路径

        返回:
            总时间（小时）= Σ(每商家无人机飞行时间) + 骑手配送时间。
        """
        drone_time = self.total_drone_time(paths)

        rider_time = 0.0
        if self.customers:
            service_points = self._extract_service_points_from_paths(paths)
            if not service_points:
                service_points = [(float(p[0]), float(p[1])) for p in self.candidate_points]

        if self.customers and service_points:
            speed_km_per_h = self.rider_speed
            for customer in self.customers:
                cx, cy = customer.get('x', 0.0), customer.get('y', 0.0)
                min_distance = float('inf')
                for px, py in service_points:
                    dist = sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if dist < min_distance:
                        min_distance = dist
                rider_time += (min_distance / 1000.0) / speed_km_per_h

        return float(drone_time + rider_time)

    def total_cost(
        self,
        paths: List[List[List[Tuple[float, float, float]]]],
    ) -> float:
        """计算所有商家到所有候选点路径的总成本。

        参数:
            paths: 路径列表，paths[i][j] 对应 merchants[i] 到候选点 j 的路径

        返回:
            Σ(每条路径的: 地形代价 + 障碍物代价 + 路程代价 + 高度变化代价 + 转角代价) + 配送时间代价。
        """
        if not paths:
            return 0.0

        total = 0.0
        for i, merchant_paths in enumerate(paths):
            if i >= len(self.merchants):
                break

            merchant = self.merchants[i]
            mx, my = merchant.get('x', 0.0), merchant.get('y', 0.0)
            mz = self.terrain.get_elevation(mx, my)

            for path in merchant_paths:
                start = (mx, my, mz)
                end = (path[-1][0], path[-1][1], path[-1][2]) if path else start

                tc = self.terrain_cost(path)
                oc = self.obstacle_collision_cost(path, start, end)
                fc = self.flight_distance_cost(path, start, end)
                ac = self.altitude_variation_cost(path, start, end)
                angc = self.turning_angle_cost(path, start, end)
                total += tc + oc + fc + ac + angc

        rc = self.rider_time_cost(paths)
        return float(total + rc)


def _ensure_required_columns(
    actual_columns: Sequence[str], required_columns: Sequence[str], csv_path: Path
) -> None:
    """校验 CSV 是否包含必需字段。

    参数:
        actual_columns: CSV 实际字段名列表
        required_columns: 必需字段名列表
        csv_path: CSV 文件路径（用于报错信息）

    异常:
        ValueError: 当缺少必需字段时抛出。
    """

    missing = [col for col in required_columns if col not in actual_columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}. "
            f"Available columns: {list(actual_columns)}"
        )


def load_terrain_from_csv(csv_path: str) -> Terrain:
    """从地形 CSV 构建 Terrain 对象。

    参数:
        csv_path: 地形数据路径，需包含列 Row, Col, X, Y, T_elevation

    返回:
        Terrain 对象

    说明:
        - 自动识别网格尺寸
        - 若有缺失格点，会用邻域均值（或全局均值）填补。
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Terrain CSV not found: {path}")

    samples: List[Tuple[int, int, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        _ensure_required_columns(reader.fieldnames, ["Row", "Col", "X", "Y", "T_elevation"], path)

        for line_no, row in enumerate(reader, start=2):
            try:
                r = int(float(row["Row"]))
                c = int(float(row["Col"]))
                x = float(row["X"])
                y = float(row["Y"])
                h = float(row["T_elevation"])
            except Exception as exc:
                raise ValueError(f"Invalid terrain record at line {line_no}: {row}") from exc
            samples.append((r, c, x, y, h))

    if not samples:
        raise ValueError(f"{path} has no terrain data rows")

    row_ids = sorted({r for r, _, _, _, _ in samples})
    col_ids = sorted({c for _, c, _, _, _ in samples})
    row_to_idx = {r: i for i, r in enumerate(row_ids)}
    col_to_idx = {c: i for i, c in enumerate(col_ids)}

    ny = len(row_ids)
    nx = len(col_ids)

    elev_sum = np.zeros((ny, nx), dtype=float)
    elev_count = np.zeros((ny, nx), dtype=int)

    x_sum: Dict[int, float] = {c: 0.0 for c in col_ids}
    x_count: Dict[int, int] = {c: 0 for c in col_ids}
    y_sum: Dict[int, float] = {r: 0.0 for r in row_ids}
    y_count: Dict[int, int] = {r: 0 for r in row_ids}

    for r, c, x, y, h in samples:
        i = row_to_idx[r]
        j = col_to_idx[c]
        elev_sum[i, j] += h
        elev_count[i, j] += 1

        x_sum[c] += x
        x_count[c] += 1
        y_sum[r] += y
        y_count[r] += 1

    elevation = np.full((ny, nx), np.nan, dtype=float)
    valid = elev_count > 0
    elevation[valid] = elev_sum[valid] / elev_count[valid]

    x_coords = np.array([x_sum[c] / x_count[c] for c in col_ids], dtype=float)
    y_coords = np.array([y_sum[r] / y_count[r] for r in row_ids], dtype=float)

    x_order = np.argsort(x_coords)
    y_order = np.argsort(y_coords)

    x_coords = x_coords[x_order]
    y_coords = y_coords[y_order]
    elevation = elevation[np.ix_(y_order, x_order)]

    if np.isnan(elevation).any():
        global_mean = float(np.nanmean(elevation))
        if np.isnan(global_mean):
            raise ValueError(f"All terrain elevations are NaN in {path}")

        nan_positions = np.argwhere(np.isnan(elevation))
        for iy, ix in nan_positions:
            neighbors = []
            if iy > 0 and not np.isnan(elevation[iy - 1, ix]):
                neighbors.append(elevation[iy - 1, ix])
            if iy < elevation.shape[0] - 1 and not np.isnan(elevation[iy + 1, ix]):
                neighbors.append(elevation[iy + 1, ix])
            if ix > 0 and not np.isnan(elevation[iy, ix - 1]):
                neighbors.append(elevation[iy, ix - 1])
            if ix < elevation.shape[1] - 1 and not np.isnan(elevation[iy, ix + 1]):
                neighbors.append(elevation[iy, ix + 1])
            elevation[iy, ix] = float(np.mean(neighbors)) if neighbors else global_mean

    return Terrain(elevation_matrix=elevation, x_coords=x_coords, y_coords=y_coords)


def load_obstacles_from_csv(csv_path: str, min_radius: float = 0.0) -> List[Obstacle]:
    """从建筑物 CSV 构建障碍物列表。

    参数:
        csv_path: 建筑数据路径，需包含列 Center_X, Center_Y, Elevation, Radius_m
        min_radius: 最小半径阈值，半径 <= 该值的记录会被过滤

    返回:
        Obstacle 列表

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 表头缺失、字段非法或无可用障碍物
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Building CSV not found: {path}")

    obstacles: List[Obstacle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        _ensure_required_columns(reader.fieldnames, ["Center_X", "Center_Y", "Elevation", "Radius_m"], path)

        for line_no, row in enumerate(reader, start=2):
            try:
                x = float(row["Center_X"])
                y = float(row["Center_Y"])
                z = float(row["Elevation"])
                r = float(row["Radius_m"])
            except Exception as exc:
                raise ValueError(f"Invalid obstacle record at line {line_no}: {row}") from exc

            if r <= min_radius:
                continue

            obstacles.append(Obstacle(x=x, y=y, z=z, r=r))

    if not obstacles:
        raise ValueError(
            f"No valid obstacles loaded from {path}. "
            "Check Radius_m and other numeric fields."
        )

    return obstacles


def load_candidates_from_csv(csv_path: str) -> List[Tuple[float, float]]:
    """从候选点 CSV 加载候选点坐标。

    参数:
        csv_path: 候选点 CSV 文件路径

    返回:
        候选点坐标列表，每项为 (X, Y) 元组。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    candidates = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["X"])
            y = float(row["Y"])
            candidates.append((x, y))

    return candidates


def build_calculator_from_csv(
    terrain_csv_path: str,
    building_csv_path: str,
    candidate_csv_path: str,
    **calculator_kwargs: Any,
) -> UAVPathCostCalculator:
    """从 CSV 一步构建 UAVPathCostCalculator。

    参数:
        terrain_csv_path: 地形 CSV 路径
        building_csv_path: 建筑 CSV 路径
        candidate_csv_path: 候选点 CSV 路径
        - **calculator_kwargs: 透传给 UAVPathCostCalculator 的参数

    返回:
        初始化完成的 UAVPathCostCalculator 实例。
    """

    terrain = load_terrain_from_csv(terrain_csv_path)
    obstacles = load_obstacles_from_csv(building_csv_path)
    candidates = load_candidates_from_csv(candidate_csv_path)
    return UAVPathCostCalculator(terrain=terrain, obstacles=obstacles, candidate_points=candidates, **calculator_kwargs)


def _default_data_paths() -> Tuple[Path, Path, Path]:
    """返回项目默认数据路径。"""

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "数据"
    candidate_dir = project_root / "数据处理"
    terrain_csv = data_dir / "投影加剪裁后的地形高度数据.csv"
    building_csv = data_dir / "裁剪后的建筑物数据_简洁版.csv"
    candidate_csv = candidate_dir / "候选点_适中.csv"
    
    return terrain_csv, building_csv, candidate_csv


def main() -> None:
    """模块演示入口。

    逻辑:
        1. 若默认数据文件存在，则加载真实数据并计算示例路径代价。
        2. 否则回退到随机模拟数据，保证脚本可独立运行。

    说明:
        当前演示使用多商家到所有候选点的路径结构:
        - merchants: 商家列表
        - paths[i][j]: 商家 merchants[i] 到候选点 candidate_points[j] 的路径
        - 每个商家的无人机可飞到任意候选点
        - rider_time_cost 基于候选点计算
    """

    terrain_csv, building_csv, candidate_csv = _default_data_paths()

    if terrain_csv.exists() and building_csv.exists() and candidate_csv.exists():
        terrain = load_terrain_from_csv(str(terrain_csv))
        obstacles = load_obstacles_from_csv(str(building_csv))
        candidate_points = load_candidates_from_csv(str(candidate_csv))

        print("Loaded real dataset successfully.")
        print(f"Terrain grid: {terrain.ny} x {terrain.nx}")
        print(f"Obstacle count: {len(obstacles)}")
        print(f"Candidate points count: {len(candidate_points)}")
        print(f"X range: [{terrain.x_min:.3f}, {terrain.x_max:.3f}]")
        print(f"Y range: [{terrain.y_min:.3f}, {terrain.y_max:.3f}]")
        print()
        print("Note: 真实路径规划需要:")
        print("  1. 使用 select_random_merchants() 选择商家")
        print("  2. 使用 select_random_customers() 选择顾客")
        print("  3. 优化算法生成 paths (List[List[List[Tuple[float,float,float]]]])")
        print("     其中 paths[i][j] 是商家i到候选点j的路径")
        print("  4. 调用 calculator.total_cost(paths) 评估代价")
    else:
        np.random.seed(42)
        demo_elevation = np.random.uniform(0, 50, size=(128, 128))
        terrain = Terrain(demo_elevation, x_range=(0, 1000), y_range=(0, 1000))

        obstacles = [
            Obstacle(x=300, y=400, z=30, r=25),
            Obstacle(x=600, y=200, z=45, r=30),
            Obstacle(x=800, y=700, z=20, r=20),
        ]

        merchants = [
            {'name': 'DemoMerchant1', 'x': 100.0, 'y': 100.0},
            {'name': 'DemoMerchant2', 'x': 500.0, 'y': 200.0},
        ]
        customers = [
            {'name': 'Customer1', 'x': 200.0, 'y': 150.0},
            {'name': 'Customer2', 'x': 400.0, 'y': 350.0},
            {'name': 'Customer3', 'x': 700.0, 'y': 600.0},
        ]

        calculator = UAVPathCostCalculator(
            terrain, obstacles,
            merchants=merchants,
            customers=customers,
            c_delivery=1.0,
            rider_speed=15.0,
            drone_speed=50.0,
        )

        paths = [
            [
                [(150.0, 150.0, 60.0), (200.0, 200.0, 70.0)],
                [(300.0, 250.0, 65.0), (400.0, 350.0, 75.0)],
            ],
            [
                [(550.0, 250.0, 60.0), (600.0, 300.0, 70.0)],
                [(700.0, 400.0, 65.0), (800.0, 500.0, 75.0)],
            ],
        ]

        total_cost = calculator.total_cost(paths)
        total_time = calculator.total_time(paths)

        print("Fallback to demo data.")
        print(f"Merchant count: {len(merchants)}")
        print(f"Customer count: {len(customers)}")
        print(f"Total cost: {total_cost:.2f}")
        print(f"Total time: {total_time:.4f} hours")


def select_random_merchants(
    csv_path: str,
    n: int,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从商家数据CSV中随机抽取n个商家作为模型起点。

    参数:
        csv_path: 商家数据CSV文件路径
        n: 抽取的商家数量
        random_seed: 随机种子，用于复现结果

    返回:
        包含商家信息的字典列表，每条记录包含:
        - id: 商家ID
        - name: 商家名称
        - address: 商家地址
        - phone: 商家电话
        - x: Center_X 投影坐标
        - y: Center_Y 投影坐标
        - longitude: 经度_GCJ02
        - latitude: 纬度_GCJ02
        - type_code: 类型代码
        - type_name: 类型名称
        - rating: 评分

    异常:
        ValueError: 当 n 大于CSV总行数时抛出
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if n > len(rows):
        raise ValueError(f"Requested {n} merchants but only {len(rows)} available in CSV")

    selected_indices = np.random.choice(len(rows), size=n, replace=False)

    result = []
    for idx in selected_indices:
        row = rows[idx]
        x_str = row.get('Center_X', '0')
        y_str = row.get('Center_Y', '0')
        rating_str = row.get('评分', '')

        result.append({
            'id': row.get('id', ''),
            'name': row.get('名称', ''),
            'address': row.get('地址', ''),
            'phone': row.get('电话', ''),
            'x': float(x_str) if x_str else 0.0,
            'y': float(y_str) if y_str else 0.0,
            'longitude': row.get('经度_GCJ02', ''),
            'latitude': row.get('纬度_GCJ02', ''),
            'type_code': row.get('类型代码', ''),
            'type_name': row.get('类型名称', ''),
            'rating': float(rating_str) if rating_str and rating_str != '[]' else None,
        })

    return result


def select_random_customers(
    csv_path: str,
    m: int,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从顾客数据CSV中随机抽取m个顾客。

    参数:
        csv_path: 顾客数据CSV文件路径
        m: 抽取的顾客数量
        random_seed: 随机种子，用于复现结果

    返回:
        包含顾客信息的字典列表，每条记录包含:
        - id: 顾客ID
        - name: 顾客名称
        - address: 顾客地址
        - phone: 顾客电话
        - x: Center_X 投影坐标
        - y: Center_Y 投影坐标
        - longitude: 经度_GCJ02
        - latitude: 纬度_GCJ02
        - type_code: 类型代码
        - type_name: 类型名称
        - rating: 评分

    异常:
        ValueError: 当 m 大于CSV总行数时抛出
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if m > len(rows):
        raise ValueError(f"Requested {m} customers but only {len(rows)} available in CSV")

    selected_indices = np.random.choice(len(rows), size=m, replace=False)

    result = []
    for idx in selected_indices:
        row = rows[idx]
        x_str = row.get('Center_X', '0')
        y_str = row.get('Center_Y', '0')
        rating_str = row.get('评分', '')

        result.append({
            'id': row.get('id', ''),
            'name': row.get('名称', ''),
            'address': row.get('地址', ''),
            'phone': row.get('电话', ''),
            'x': float(x_str) if x_str else 0.0,
            'y': float(y_str) if y_str else 0.0,
            'longitude': row.get('经度_GCJ02', ''),
            'latitude': row.get('纬度_GCJ02', ''),
            'type_code': row.get('类型代码', ''),
            'type_name': row.get('类型名称', ''),
            'rating': float(rating_str) if rating_str and rating_str != '[]' else None,
        })

    return result


@dataclass
class DeliveryOrder:
    """融合模型中的订单实体。"""

    order_id: str
    merchant_idx: int
    customer_idx: int
    customer_x: float
    customer_y: float
    demand: float = 1.0
    earliest_time: float = 0.0
    latest_time: float = 1.5
    service_time: float = 0.03  # 小时，约 1.8 分钟


@dataclass
class FusionModelConfig:
    """无人机-骑手协同融合模型参数。"""

    max_selected_candidates: int = 4
    candidate_capacity: float = float("inf")
    drone_capacity: float = 3.0
    drone_energy_limit: float = float("inf")
    rider_capacity: float = 20.0
    drone_turnaround_time: float = 0.03
    safe_clearance_height: float = 20.0
    min_flight_height: float = 20.0
    max_flight_height: float = 120.0
    alpha_spatial: float = 0.6
    late_penalty_coeff: float = 10.0
    candidate_opening_cost: float = 1.0
    lambda_drone_cost: float = 1.0
    lambda_rider_cost: float = 1.0
    lambda_late_cost: float = 1.0
    lambda_open_cost: float = 1.0
    lambda_makespan: float = 1.0
    select_improvement_tolerance: float = 1e-9
    allow_infeasible_fallback: bool = False


@dataclass
class FusedModelSolution:
    """融合优化结果。"""

    selected_candidates: List[int]
    order_plan: List[Dict[str, Any]]
    drone_order_indices: Dict[int, List[int]]
    rider_order_indices: Dict[int, List[int]]
    objective_components: Dict[str, float]
    total_objective: float
    constraint_report: Dict[str, Any]


def _terrain_color_map_vectorized(z_normalized: np.ndarray) -> np.ndarray:
    """仿照 render_map.py 的分段渐变地形配色。"""

    z_norm = np.asarray(z_normalized, dtype=float)
    z = np.clip(z_norm.reshape(-1), 0.0, 1.0)
    r = np.zeros_like(z)
    g = np.zeros_like(z)
    b = np.zeros_like(z)

    mask1 = z < 0.2
    mask2 = (z >= 0.2) & (z < 0.4)
    mask3 = (z >= 0.4) & (z < 0.6)
    mask4 = (z >= 0.6) & (z < 0.8)
    mask5 = z >= 0.8

    t1 = z[mask1] / 0.2
    r[mask1] = 0.20 + 0.10 * t1
    g[mask1] = 0.50 + 0.20 * t1
    b[mask1] = 0.25 + 0.05 * t1

    t2 = (z[mask2] - 0.2) / 0.2
    r[mask2] = 0.30 + 0.15 * t2
    g[mask2] = 0.70 + 0.05 * t2
    b[mask2] = 0.30 - 0.10 * t2

    t3 = (z[mask3] - 0.4) / 0.2
    r[mask3] = 0.45 + 0.15 * t3
    g[mask3] = 0.75 - 0.15 * t3
    b[mask3] = 0.20 + 0.05 * t3

    t4 = (z[mask4] - 0.6) / 0.2
    r[mask4] = 0.60 + 0.15 * t4
    g[mask4] = 0.60 - 0.20 * t4
    b[mask4] = 0.25 + 0.10 * t4

    t5 = (z[mask5] - 0.8) / 0.2
    r[mask5] = 0.75 + 0.15 * t5
    g[mask5] = 0.40 - 0.15 * t5
    b[mask5] = 0.35 + 0.15 * t5

    colors = np.stack([r, g, b], axis=1)
    return colors.reshape((*z_norm.shape, 3))


def _building_color_by_height(height: float) -> Tuple[float, float, float]:
    """仿照 render_map.py 的建筑高度分级配色。"""

    if height < 50.0:
        return (0.40, 0.42, 0.45)
    if height < 100.0:
        return (0.50, 0.55, 0.62)
    if height < 150.0:
        return (0.60, 0.65, 0.72)
    if height < 200.0:
        return (0.72, 0.76, 0.82)
    return (0.85, 0.87, 0.90)


def _extract_solution_routes(
    terrain: Terrain,
    merchants: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    candidate_points: List[Tuple[float, float]],
    solution: FusedModelSolution,
    safe_clearance_height: float,
    min_flight_height: float,
    max_flight_height: float,
    rider_ground_lift: float,
) -> List[Dict[str, Any]]:
    """将最终规划结果转换为可绘制的无人机/骑手路径。"""

    routes: List[Dict[str, Any]] = []
    if not candidate_points or not solution.order_plan:
        return routes

    lift = max(0.5, float(rider_ground_lift))
    for plan in solution.order_plan:
        merchant_idx = int(plan.get("merchant_idx", -1))
        customer_idx = int(plan.get("customer_idx", -1))
        candidate_idx = int(plan.get("candidate_idx", -1))
        drone_idx = int(plan.get("drone_idx", -1))
        rider_idx = int(plan.get("rider_idx", -1))

        if not (0 <= merchant_idx < len(merchants)):
            continue
        if not (0 <= candidate_idx < len(candidate_points)):
            continue

        merchant = merchants[merchant_idx]
        mx = float(merchant.get("x", 0.0))
        my = float(merchant.get("y", 0.0))
        gx, gy = candidate_points[candidate_idx]
        gx = float(gx)
        gy = float(gy)

        if 0 <= customer_idx < len(customers):
            customer = customers[customer_idx]
            cx = float(customer.get("x", gx))
            cy = float(customer.get("y", gy))
        else:
            cx, cy = gx, gy

        mz = terrain.get_elevation(mx, my)
        gz = terrain.get_elevation(gx, gy)
        cz = terrain.get_elevation(cx, cy)

        cruise = max(mz, gz) + float(safe_clearance_height)
        cruise = float(np.clip(cruise, float(min_flight_height), float(max_flight_height)))

        mid = ((mx + gx) * 0.5, (my + gy) * 0.5, cruise)
        uav_path = [(mx, my, mz), mid, (gx, gy, cruise)]
        rider_path = [(gx, gy, gz + lift), (cx, cy, cz + lift)]

        routes.append(
            {
                "order_index": int(plan.get("order_index", -1)),
                "drone_idx": drone_idx,
                "rider_idx": rider_idx,
                "candidate_idx": candidate_idx,
                "uav_path": uav_path,
                "rider_path": rider_path,
            }
        )
    return routes


def render_fused_solution_map(
    terrain: Terrain,
    obstacles: List[Obstacle],
    merchants: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    candidate_points: List[Tuple[float, float]],
    solution: FusedModelSolution,
    save_path: Optional[str] = None,
    show: bool = True,
    safe_clearance_height: float = 20.0,
    min_flight_height: float = 20.0,
    max_flight_height: float = 120.0,
    rider_ground_lift: float = 2.0,
    window_size: Tuple[int, int] = (1600, 1000),
) -> Optional[str]:
    """绘制融合模型最终规划路径图（地形+建筑+无人机+骑手）。

    风格参考 render_map.py，包含:
    - 分段地形渐变色
    - 建筑按高度分级配色
    - 柔和背景与双光源
    """

    if not show and not save_path:
        raise ValueError("show=False 时需要提供 save_path 以便输出图像。")

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "render_fused_solution_map 依赖 pyvista，请先安装: pip install pyvista"
        ) from exc

    grid_x, grid_y = np.meshgrid(terrain.x_coords, terrain.y_coords)
    grid_z = np.asarray(terrain.elevation, dtype=float)

    terrain_mesh = pv.StructuredGrid(grid_x, grid_y, grid_z)
    z_min = float(np.nanmin(grid_z))
    z_max = float(np.nanmax(grid_z))
    z_norm = (grid_z - z_min) / (z_max - z_min + 1e-9)
    terrain_colors = _terrain_color_map_vectorized(z_norm)
    terrain_mesh["terrain_colors"] = terrain_colors.reshape(-1, 3)

    off_screen = bool(save_path) and not show
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)

    plotter.add_mesh(
        terrain_mesh,
        rgb=True,
        scalars="terrain_colors",
        show_scalar_bar=False,
        opacity=0.92,
        show_edges=False,
        edge_color=(0.3, 0.4, 0.3),
        smooth_shading=True,
    )

    for obs in obstacles:
        radius = float(obs.r)
        height = float(obs.z)
        if radius < 1.0 or height <= 0.0:
            continue

        ox = float(obs.x)
        oy = float(obs.y)
        base_z = terrain.get_elevation(ox, oy)
        building = pv.Cylinder(
            center=(ox, oy, base_z + height / 2.0),
            direction=(0.0, 0.0, 1.0),
            radius=radius,
            height=height,
            resolution=12,
        )
        plotter.add_mesh(
            building,
            color=_building_color_by_height(height),
            opacity=0.95,
            smooth_shading=True,
            specular=0.4,
            metallic=0.15,
        )

    routes = _extract_solution_routes(
        terrain=terrain,
        merchants=merchants,
        customers=customers,
        candidate_points=candidate_points,
        solution=solution,
        safe_clearance_height=safe_clearance_height,
        min_flight_height=min_flight_height,
        max_flight_height=max_flight_height,
        rider_ground_lift=rider_ground_lift,
    )

    drone_palette = [
        (0.94, 0.35, 0.24),
        (0.96, 0.58, 0.20),
        (0.86, 0.44, 0.24),
        (0.90, 0.66, 0.26),
        (0.78, 0.33, 0.21),
    ]
    rider_palette = [
        (0.16, 0.46, 0.76),
        (0.21, 0.58, 0.67),
        (0.23, 0.52, 0.44),
        (0.25, 0.40, 0.72),
        (0.13, 0.34, 0.57),
    ]

    for route in routes:
        uav_path = np.asarray(route["uav_path"], dtype=float)
        rider_path = np.asarray(route["rider_path"], dtype=float)
        drone_idx = int(route["drone_idx"])
        rider_idx = int(route["rider_idx"])

        drone_color = drone_palette[drone_idx % len(drone_palette)] if drone_idx >= 0 else drone_palette[0]
        rider_color = rider_palette[rider_idx % len(rider_palette)] if rider_idx >= 0 else rider_palette[0]

        if uav_path.shape[0] >= 3:
            drone_line = pv.Spline(uav_path, n_points=80)
        else:
            drone_line = pv.Line(uav_path[0], uav_path[-1], resolution=1)
        plotter.add_mesh(
            drone_line,
            color=drone_color,
            line_width=5.0,
            render_lines_as_tubes=True,
            opacity=0.95,
        )

        rider_line = pv.Line(rider_path[0], rider_path[-1], resolution=1)
        plotter.add_mesh(
            rider_line,
            color=rider_color,
            line_width=3.0,
            render_lines_as_tubes=True,
            opacity=0.90,
        )

    if merchants:
        merchant_xyz = np.array(
            [
                (
                    float(m.get("x", 0.0)),
                    float(m.get("y", 0.0)),
                    terrain.get_elevation(float(m.get("x", 0.0)), float(m.get("y", 0.0))) + 3.0,
                )
                for m in merchants
            ],
            dtype=float,
        )
        plotter.add_points(
            merchant_xyz,
            color=(0.82, 0.20, 0.16),
            point_size=15,
            render_points_as_spheres=True,
        )

    if customers:
        customer_xyz = np.array(
            [
                (
                    float(c.get("x", 0.0)),
                    float(c.get("y", 0.0)),
                    terrain.get_elevation(float(c.get("x", 0.0)), float(c.get("y", 0.0))) + 2.0,
                )
                for c in customers
            ],
            dtype=float,
        )
        plotter.add_points(
            customer_xyz,
            color=(0.16, 0.42, 0.70),
            point_size=11,
            render_points_as_spheres=True,
        )

    if candidate_points:
        candidate_xyz = np.array(
            [
                (float(gx), float(gy), terrain.get_elevation(float(gx), float(gy)) + 1.5)
                for gx, gy in candidate_points
            ],
            dtype=float,
        )
        selected_set = set(int(i) for i in solution.selected_candidates)
        selected_idx = [i for i in range(len(candidate_points)) if i in selected_set]
        unselected_idx = [i for i in range(len(candidate_points)) if i not in selected_set]

        if unselected_idx:
            plotter.add_points(
                candidate_xyz[unselected_idx, :],
                color=(0.94, 0.94, 0.94),
                point_size=8,
                render_points_as_spheres=True,
                opacity=0.75,
            )
        if selected_idx:
            plotter.add_points(
                candidate_xyz[selected_idx, :],
                color=(0.98, 0.82, 0.22),
                point_size=16,
                render_points_as_spheres=True,
            )

    plotter.set_background("#d0dde8")

    center_x = 0.5 * (terrain.x_min + terrain.x_max)
    center_y = 0.5 * (terrain.y_min + terrain.y_max)
    center_z = float(np.nanmean(grid_z))

    light1 = pv.Light(
        position=(terrain.x_min - 300.0, terrain.y_min - 300.0, z_max + 300.0),
        focal_point=(center_x, center_y, center_z),
        intensity=1.0,
        color=(1.0, 0.98, 0.95),
    )
    plotter.add_light(light1)

    light2 = pv.Light(
        position=(terrain.x_max + 300.0, terrain.y_max + 300.0, z_max + 200.0),
        focal_point=(center_x, center_y, center_z),
        intensity=0.5,
        color=(0.9, 0.92, 1.0),
    )
    plotter.add_light(light2)
    plotter.enable_lightkit()

    plotter.camera_position = [
        (terrain.x_min - 300.0, terrain.y_min - 300.0, z_max + 250.0),
        (center_x, center_y, center_z),
        (0.0, 0.0, 1.0),
    ]

    plotter.add_text(
        "Fused Delivery Plan",
        position="upper_left",
        font_size=12,
        color=(0.12, 0.18, 0.26),
    )

    saved_file: Optional[str] = None
    if save_path:
        save_file = Path(save_path).expanduser().resolve()
        save_file.parent.mkdir(parents=True, exist_ok=True)
        saved_file = str(save_file)

    if show:
        if saved_file:
            plotter.show(screenshot=saved_file)
        else:
            plotter.show()
    else:
        plotter.show(auto_close=False)
        if saved_file:
            plotter.screenshot(saved_file)
        plotter.close()

    return saved_file


class InfeasibleFusionPlanError(RuntimeError):
    """融合模型不可行时抛出的异常。"""


def build_orders_from_customers(
    customers: List[Dict[str, Any]],
    merchants: List[Dict[str, Any]],
    order_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    demand_range: Tuple[float, float] = (0.5, 1.5),
    earliest_range: Tuple[float, float] = (0.0, 0.5),
    window_width: float = 1.0,
    service_time: float = 0.03,
) -> List[DeliveryOrder]:
    """从顾客样本构建订单集合（用于融合模型求解）。

    说明:
        - 若顾客数据中缺少订单量/时间窗，本函数自动合成可复现实验订单。
        - 每个顾客默认生成 1 个订单。
        - 订单归属商家采用“最近商家”原则。
    """

    if not merchants:
        raise ValueError("merchants is empty; cannot infer merchant_idx for orders")
    if not customers:
        return []

    rng = np.random.default_rng(random_seed)
    count = len(customers) if order_count is None else min(order_count, len(customers))
    indices = np.arange(len(customers))
    if count < len(customers):
        indices = rng.choice(indices, size=count, replace=False)

    def _nearest_merchant_idx(cx: float, cy: float) -> int:
        best_idx = 0
        best_dist2 = float("inf")
        for mi, merchant in enumerate(merchants):
            mx = float(merchant.get("x", 0.0))
            my = float(merchant.get("y", 0.0))
            d2 = (cx - mx) ** 2 + (cy - my) ** 2
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = mi
        return best_idx

    low_demand, high_demand = demand_range
    early_low, early_high = earliest_range

    orders: List[DeliveryOrder] = []
    for k, idx in enumerate(indices):
        c = customers[int(idx)]
        cx = float(c.get("x", 0.0))
        cy = float(c.get("y", 0.0))
        earliest = float(rng.uniform(early_low, early_high))
        latest = earliest + float(window_width)
        demand = float(rng.uniform(low_demand, high_demand))
        order_id = str(c.get("id", f"O{k}"))

        orders.append(
            DeliveryOrder(
                order_id=order_id,
                merchant_idx=_nearest_merchant_idx(cx, cy),
                customer_idx=int(idx),
                customer_x=cx,
                customer_y=cy,
                demand=demand,
                earliest_time=earliest,
                latest_time=latest,
                service_time=float(service_time),
            )
        )

    return orders


class DroneRiderFusionOptimizer:
    """候选点选择 + 订单分配 + 多无人机 + 骑手接驳的一体化求解器。

    该实现将论文中的总目标函数落地为可运行代码：
        Z = λ1*无人机成本 + λ2*骑手时空成本 + λ3*超时惩罚 + λ4*候选点启用成本 + λ5*完工时间

    说明:
        - 无需额外外部求解器，采用可复现实验的启发式求解。
        - 无人机航段代价仍调用现有 UAVPathCostCalculator 的子代价函数，保持与原模型一致。
    """

    def __init__(
        self,
        calculator: UAVPathCostCalculator,
        merchants: List[Dict[str, Any]],
        candidate_points: List[Tuple[float, float]],
        orders: List[DeliveryOrder],
        n_drones: int = 3,
        n_riders: int = 4,
        config: Optional[FusionModelConfig] = None,
    ):
        if n_drones <= 0:
            raise ValueError("n_drones must be > 0")
        if n_riders <= 0:
            raise ValueError("n_riders must be > 0")
        if not merchants:
            raise ValueError("merchants is empty")
        if not candidate_points:
            raise ValueError("candidate_points is empty")
        if not orders:
            raise ValueError("orders is empty")

        self.calculator = calculator
        self.merchants = merchants
        self.candidate_points = candidate_points
        self.orders = orders
        self.n_drones = int(n_drones)
        self.n_riders = int(n_riders)
        self.config = config if config is not None else FusionModelConfig()

        self._uav_cost: Optional[np.ndarray] = None
        self._uav_time: Optional[np.ndarray] = None
        self._uav_energy: Optional[np.ndarray] = None
        self._order_candidate_score: Optional[np.ndarray] = None

    @staticmethod
    def _distance_2d(ax: float, ay: float, bx: float, by: float) -> float:
        return float(sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    @staticmethod
    def _hours_from_distance(distance_m: float, speed_km_h: float) -> float:
        speed = max(1e-9, speed_km_h)
        return float((distance_m / 1000.0) / speed)

    def _compute_uav_leg_metrics(self, merchant_idx: int, candidate_idx: int) -> Tuple[float, float, float]:
        merchant = self.merchants[merchant_idx]
        mx = float(merchant.get("x", 0.0))
        my = float(merchant.get("y", 0.0))
        gx, gy = self.candidate_points[candidate_idx]
        gx = float(gx)
        gy = float(gy)

        terrain = self.calculator.terrain
        mz = terrain.get_elevation(mx, my)
        gz = terrain.get_elevation(gx, gy)
        cruise = max(mz, gz) + self.config.safe_clearance_height
        cruise = float(np.clip(cruise, self.config.min_flight_height, self.config.max_flight_height))

        start = (mx, my, mz)
        end = (gx, gy, cruise)
        mid = ((mx + gx) * 0.5, (my + gy) * 0.5, cruise)
        path = [mid]

        tc = self.calculator.terrain_cost(path)
        oc = self.calculator.obstacle_collision_cost(path, start, end)
        fc = self.calculator.flight_distance_cost(path, start, end)
        ac = self.calculator.altitude_variation_cost(path, start, end)
        angc = self.calculator.turning_angle_cost(path, start, end)

        cost = float(tc + oc + fc + ac + angc)
        flight_time = self._hours_from_distance(fc, self.calculator.drone_speed)
        energy_proxy = float(fc)
        return cost, flight_time, energy_proxy

    def _build_uav_matrices(self) -> None:
        nm = len(self.merchants)
        ng = len(self.candidate_points)
        uav_cost = np.zeros((nm, ng), dtype=float)
        uav_time = np.zeros((nm, ng), dtype=float)
        uav_energy = np.zeros((nm, ng), dtype=float)

        for mi in range(nm):
            for gi in range(ng):
                c, t, e = self._compute_uav_leg_metrics(mi, gi)
                uav_cost[mi, gi] = c
                uav_time[mi, gi] = t
                uav_energy[mi, gi] = e

        self._uav_cost = uav_cost
        self._uav_time = uav_time
        self._uav_energy = uav_energy

    def _build_order_candidate_score_matrix(self) -> None:
        if self._uav_cost is None:
            self._build_uav_matrices()

        no = len(self.orders)
        ng = len(self.candidate_points)
        score = np.zeros((no, ng), dtype=float)

        for oi, order in enumerate(self.orders):
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0

            for gi, (gx, gy) in enumerate(self.candidate_points):
                gx = float(gx)
                gy = float(gy)
                rider_dist = self._distance_2d(order.customer_x, order.customer_y, gx, gy)
                score[oi, gi] = float(
                    self.config.lambda_drone_cost * self._uav_cost[mi, gi]
                    + self.config.lambda_rider_cost * rider_dist
                )

        self._order_candidate_score = score

    def select_candidates_greedy(self) -> List[int]:
        """按总成本下降幅度进行候选点贪心选择。"""

        if self._order_candidate_score is None:
            self._build_order_candidate_score_matrix()

        assert self._order_candidate_score is not None
        no, ng = self._order_candidate_score.shape
        max_pick = max(1, min(self.config.max_selected_candidates, ng))

        selected: List[int] = []
        remaining = set(range(ng))
        current_best = np.full(no, np.inf, dtype=float)
        current_total = float("inf")

        for _ in range(max_pick):
            best_g = None
            best_total = float("inf")
            best_vector = None

            for g in list(remaining):
                v = np.minimum(current_best, self._order_candidate_score[:, g])
                total = float(np.sum(v)) + (
                    self.config.lambda_open_cost
                    * self.config.candidate_opening_cost
                    * float(len(selected) + 1)
                )
                if total < best_total:
                    best_total = total
                    best_g = g
                    best_vector = v

            if best_g is None:
                break

            if best_total + self.config.select_improvement_tolerance >= current_total:
                break

            selected.append(best_g)
            remaining.remove(best_g)
            if best_vector is not None:
                current_best = best_vector
            current_total = best_total

        if not selected:
            selected = [0]
        return selected

    def _assign_candidates(self, selected_candidates: List[int]) -> Tuple[List[int], Dict[int, float]]:
        if self._order_candidate_score is None:
            self._build_order_candidate_score_matrix()
        assert self._order_candidate_score is not None

        candidate_load = {g: 0.0 for g in selected_candidates}
        assigned_candidate = [-1] * len(self.orders)

        for oi, order in enumerate(self.orders):
            best_g = None
            best_score = float("inf")
            fallback_g = selected_candidates[0] if selected_candidates else -1

            for g in selected_candidates:
                s = float(self._order_candidate_score[oi, g])
                if candidate_load[g] + order.demand <= self.config.candidate_capacity and s < best_score:
                    best_score = s
                    best_g = g

            if best_g is None:
                if not self.config.allow_infeasible_fallback:
                    raise InfeasibleFusionPlanError(
                        f"No feasible candidate for order {order.order_id}: "
                        "candidate capacity is insufficient under current selected set."
                    )
                chosen = fallback_g
            else:
                chosen = best_g

            assigned_candidate[oi] = chosen
            candidate_load[chosen] += float(order.demand)

        return assigned_candidate, candidate_load

    def _assign_drones(
        self,
        assigned_candidate: List[int],
    ) -> Tuple[List[int], List[float], List[float], Dict[int, List[int]], List[float]]:
        assert self._uav_cost is not None and self._uav_time is not None and self._uav_energy is not None

        n_orders = len(self.orders)
        assigned_drone = [-1] * n_orders
        depart_time = [0.0] * n_orders
        arrival_time = [0.0] * n_orders

        drone_available = [0.0] * self.n_drones
        drone_energy_used = [0.0] * self.n_drones
        drone_order_indices: Dict[int, List[int]] = defaultdict(list)

        sorted_orders = sorted(range(n_orders), key=lambda oi: self.orders[oi].earliest_time)
        for oi in sorted_orders:
            order = self.orders[oi]
            g = assigned_candidate[oi]
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0

            best_d = 0
            best_finish = float("inf")
            best_depart = 0.0
            best_arrive = 0.0
            best_energy = 0.0
            feasible_found = False

            for d in range(self.n_drones):
                if order.demand > self.config.drone_capacity:
                    continue
                e_need = float(self._uav_energy[mi, g])
                if drone_energy_used[d] + e_need > self.config.drone_energy_limit:
                    continue

                dep = max(drone_available[d], order.earliest_time)
                arr = dep + float(self._uav_time[mi, g])
                finish = arr + self.config.drone_turnaround_time
                if finish < best_finish:
                    best_finish = finish
                    best_d = d
                    best_depart = dep
                    best_arrive = arr
                    best_energy = e_need
                    feasible_found = True

            if not feasible_found:
                if not self.config.allow_infeasible_fallback:
                    raise InfeasibleFusionPlanError(
                        f"No feasible drone for order {order.order_id}: "
                        "drone capacity/energy constraints are violated."
                    )
                # 若允许回退，则给出最早可用无人机近似安排
                best_d = int(np.argmin(drone_available))
                best_depart = max(drone_available[best_d], order.earliest_time)
                best_arrive = best_depart + float(self._uav_time[mi, g])
                best_finish = best_arrive + self.config.drone_turnaround_time
                best_energy = float(self._uav_energy[mi, g])

            assigned_drone[oi] = best_d
            depart_time[oi] = best_depart
            arrival_time[oi] = best_arrive

            drone_available[best_d] = best_finish
            drone_energy_used[best_d] += best_energy
            drone_order_indices[best_d].append(oi)

        return assigned_drone, depart_time, arrival_time, drone_order_indices, drone_energy_used

    def _temporal_distance_component(self, order_i: DeliveryOrder, order_j: DeliveryOrder) -> float:
        """按论文分段公式近似计算 d_ij^T（单位：小时）。"""

        dist_ij = self._distance_2d(
            order_i.customer_x, order_i.customer_y, order_j.customer_x, order_j.customer_y
        )
        t_ij = self._hours_from_distance(dist_ij, self.calculator.rider_speed) + order_i.service_time

        lt_i = float(order_i.earliest_time)
        ut_i = float(order_i.latest_time)
        lt_j = float(order_j.earliest_time)
        ut_j = float(order_j.latest_time)

        lt_j_prime = lt_i + t_ij
        ut_j_prime = ut_i + t_ij

        if ut_j_prime < lt_j:
            d_t = 0.5 * (2.0 * lt_j - ut_j_prime - lt_j_prime) + t_ij
        elif lt_j_prime < lt_j <= ut_j_prime:
            d_t = 0.5 * (lt_j - lt_j_prime) + t_ij
        elif lt_j <= lt_j_prime and ut_j_prime <= ut_j:
            d_t = t_ij
        elif lt_j_prime <= ut_j < ut_j_prime:
            denom = max(1e-9, ut_j - lt_j_prime)
            d_t = ((ut_j_prime - lt_j_prime) / denom) * t_ij
        else:
            d_t = 1e6  # 近似 +∞
        return float(d_t)

    def _spatiotemporal_distance(self, order_i: DeliveryOrder, order_j: DeliveryOrder) -> float:
        """计算 D_ij = α*d_ij + (1-α)*d_ij^T(等价里程)。"""

        d_ij = self._distance_2d(
            order_i.customer_x, order_i.customer_y, order_j.customer_x, order_j.customer_y
        )
        d_t_h = self._temporal_distance_component(order_i, order_j)
        d_t_equiv = d_t_h * self.calculator.rider_speed * 1000.0
        alpha = float(np.clip(self.config.alpha_spatial, 0.0, 1.0))
        return float(alpha * d_ij + (1.0 - alpha) * d_t_equiv)

    def _rider_transition_cost(
        self,
        prev_order_idx: Optional[int],
        current_order_idx: int,
        candidate_idx: int,
    ) -> float:
        """计算骑手从上一个服务节点转移到当前订单的时空代价。"""

        current_order = self.orders[current_order_idx]
        gx, gy = self.candidate_points[candidate_idx]
        gx = float(gx)
        gy = float(gy)

        if prev_order_idx is None:
            spatial_m = self._distance_2d(gx, gy, current_order.customer_x, current_order.customer_y)
            temporal_equiv_m = 0.0
        else:
            prev_order = self.orders[prev_order_idx]
            reposition_m = self._distance_2d(prev_order.customer_x, prev_order.customer_y, gx, gy)
            delivery_m = self._distance_2d(gx, gy, current_order.customer_x, current_order.customer_y)
            spatial_m = reposition_m + delivery_m
            temporal_h = self._temporal_distance_component(prev_order, current_order)
            temporal_equiv_m = temporal_h * self.calculator.rider_speed * 1000.0

        alpha = float(np.clip(self.config.alpha_spatial, 0.0, 1.0))
        return float(alpha * spatial_m + (1.0 - alpha) * temporal_equiv_m)

    def _assign_riders(
        self,
        assigned_candidate: List[int],
        arrival_time: List[float],
    ) -> Tuple[List[int], List[float], List[float], Dict[int, List[int]], float]:
        n_orders = len(self.orders)
        assigned_rider = [-1] * n_orders
        delivery_time = [0.0] * n_orders
        lateness = [0.0] * n_orders
        rider_order_indices: Dict[int, List[int]] = defaultdict(list)

        rider_available = [0.0] * self.n_riders
        rider_pos: List[Optional[Tuple[float, float]]] = [None] * self.n_riders
        rider_spatiotemporal_cost = 0.0

        sorted_orders = sorted(range(n_orders), key=lambda oi: arrival_time[oi])
        for oi in sorted_orders:
            order = self.orders[oi]
            g = assigned_candidate[oi]
            gx, gy = self.candidate_points[g]
            gx = float(gx)
            gy = float(gy)

            best_r = 0
            best_finish = float("inf")
            best_depart = 0.0
            feasible_found = False

            for r in range(self.n_riders):
                if order.demand > self.config.rider_capacity:
                    continue

                pos = rider_pos[r]
                to_candidate_m = 0.0 if pos is None else self._distance_2d(pos[0], pos[1], gx, gy)
                to_candidate_h = self._hours_from_distance(to_candidate_m, self.calculator.rider_speed)
                depart = max(rider_available[r] + to_candidate_h, arrival_time[oi], order.earliest_time)

                leg_m = self._distance_2d(gx, gy, order.customer_x, order.customer_y)
                leg_h = self._hours_from_distance(leg_m, self.calculator.rider_speed)
                finish = depart + leg_h + order.service_time

                if finish < best_finish:
                    best_finish = finish
                    best_r = r
                    best_depart = depart
                    feasible_found = True

            if not feasible_found:
                if not self.config.allow_infeasible_fallback:
                    raise InfeasibleFusionPlanError(
                        f"No feasible rider for order {order.order_id}: "
                        "rider capacity constraints are violated."
                    )
                best_r = int(np.argmin(rider_available))
                pos = rider_pos[best_r]
                to_candidate_m = 0.0 if pos is None else self._distance_2d(pos[0], pos[1], gx, gy)
                to_candidate_h = self._hours_from_distance(to_candidate_m, self.calculator.rider_speed)
                best_depart = max(rider_available[best_r] + to_candidate_h, arrival_time[oi], order.earliest_time)
                leg_m = self._distance_2d(gx, gy, order.customer_x, order.customer_y)
                leg_h = self._hours_from_distance(leg_m, self.calculator.rider_speed)
                best_finish = best_depart + leg_h + order.service_time

            assigned_rider[oi] = best_r
            delivery_time[oi] = best_finish
            lateness[oi] = max(0.0, best_finish - order.latest_time)

            previous_orders = rider_order_indices[best_r]
            prev_idx = previous_orders[-1] if previous_orders else None
            rider_spatiotemporal_cost += self._rider_transition_cost(prev_idx, oi, g)

            rider_order_indices[best_r].append(oi)
            rider_available[best_r] = best_finish
            rider_pos[best_r] = (order.customer_x, order.customer_y)

        return assigned_rider, delivery_time, lateness, rider_order_indices, float(rider_spatiotemporal_cost)

    def _evaluate_objective(
        self,
        selected_candidates: List[int],
        assigned_candidate: List[int],
        assigned_drone: List[int],
        delivery_time: List[float],
        lateness: List[float],
        rider_spatiotemporal_cost: float,
    ) -> Tuple[Dict[str, float], float]:
        assert self._uav_cost is not None

        drone_cost = 0.0
        for oi, order in enumerate(self.orders):
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0
            g = assigned_candidate[oi]
            drone_cost += float(self._uav_cost[mi, g])

        late_cost = self.config.late_penalty_coeff * float(sum(lateness))
        open_cost = self.config.candidate_opening_cost * float(len(selected_candidates))
        makespan = max(delivery_time) if delivery_time else 0.0

        weighted_total = (
            self.config.lambda_drone_cost * drone_cost
            + self.config.lambda_rider_cost * rider_spatiotemporal_cost
            + self.config.lambda_late_cost * late_cost
            + self.config.lambda_open_cost * open_cost
            + self.config.lambda_makespan * makespan
        )

        components = {
            "drone_cost": float(drone_cost),
            "rider_spatiotemporal_cost": float(rider_spatiotemporal_cost),
            "late_cost": float(late_cost),
            "open_cost": float(open_cost),
            "makespan": float(makespan),
            "n_selected_candidates": float(len(selected_candidates)),
            "n_assigned_drones": float(len(set(assigned_drone))),
        }
        return components, float(weighted_total)

    def _build_constraint_report(
        self,
        selected_candidates: List[int],
        assigned_candidate: List[int],
        assigned_drone: List[int],
        assigned_rider: List[int],
        drone_order_indices: Dict[int, List[int]],
        drone_energy_used: List[float],
        depart_time: List[float],
        arrival_time: List[float],
        candidate_load: Dict[int, float],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        report["candidate_limit_ok"] = len(selected_candidates) <= self.config.max_selected_candidates
        report["all_orders_have_candidate"] = all(g >= 0 for g in assigned_candidate)
        report["all_orders_have_drone"] = all(d >= 0 for d in assigned_drone)
        report["all_orders_have_rider"] = all(r >= 0 for r in assigned_rider)
        report["candidate_activation_ok"] = all(g in selected_candidates for g in assigned_candidate)
        report["candidate_capacity_ok"] = all(
            load <= self.config.candidate_capacity + 1e-9 for load in candidate_load.values()
        )

        drone_capacity_ok = True
        for oi, order in enumerate(self.orders):
            if order.demand > self.config.drone_capacity + 1e-9:
                drone_capacity_ok = False
                break
        report["drone_capacity_ok"] = drone_capacity_ok

        rider_capacity_ok = True
        for oi, order in enumerate(self.orders):
            if order.demand > self.config.rider_capacity + 1e-9:
                rider_capacity_ok = False
                break
        report["rider_capacity_ok"] = rider_capacity_ok

        report["drone_energy_ok"] = all(e <= self.config.drone_energy_limit + 1e-9 for e in drone_energy_used)

        overlap_ok = True
        for drone_idx, idxs in drone_order_indices.items():
            if not idxs:
                continue
            idxs_sorted = sorted(idxs, key=lambda i: depart_time[i])
            for pos in range(1, len(idxs_sorted)):
                prev_idx = idxs_sorted[pos - 1]
                curr_idx = idxs_sorted[pos]
                required_gap = arrival_time[prev_idx] + self.config.drone_turnaround_time
                if depart_time[curr_idx] + 1e-9 < required_gap:
                    overlap_ok = False
                    break
            if not overlap_ok:
                break
            _ = drone_idx
        report["drone_overlap_ok"] = overlap_ok

        bool_values = [v for v in report.values() if isinstance(v, bool)]
        report["all_constraints_ok"] = all(bool_values)
        return report

    def solve(self) -> FusedModelSolution:
        """执行融合求解并返回结构化结果。"""

        self._build_uav_matrices()
        self._build_order_candidate_score_matrix()

        selected_candidates = self.select_candidates_greedy()
        assigned_candidate, candidate_load = self._assign_candidates(selected_candidates)

        assigned_drone, depart_time, arrival_time, drone_order_indices, drone_energy_used = self._assign_drones(
            assigned_candidate
        )
        assigned_rider, delivery_time, lateness, rider_order_indices, rider_st_cost = self._assign_riders(
            assigned_candidate, arrival_time
        )

        components, total_objective = self._evaluate_objective(
            selected_candidates=selected_candidates,
            assigned_candidate=assigned_candidate,
            assigned_drone=assigned_drone,
            delivery_time=delivery_time,
            lateness=lateness,
            rider_spatiotemporal_cost=rider_st_cost,
        )

        constraint_report = self._build_constraint_report(
            selected_candidates=selected_candidates,
            assigned_candidate=assigned_candidate,
            assigned_drone=assigned_drone,
            assigned_rider=assigned_rider,
            drone_order_indices=drone_order_indices,
            drone_energy_used=drone_energy_used,
            depart_time=depart_time,
            arrival_time=arrival_time,
            candidate_load=candidate_load,
        )

        order_plan: List[Dict[str, Any]] = []
        for oi, order in enumerate(self.orders):
            order_plan.append(
                {
                    "order_index": oi,
                    "order_id": order.order_id,
                    "merchant_idx": int(order.merchant_idx),
                    "customer_idx": int(order.customer_idx),
                    "candidate_idx": int(assigned_candidate[oi]),
                    "drone_idx": int(assigned_drone[oi]),
                    "rider_idx": int(assigned_rider[oi]),
                    "uav_depart_time_h": float(depart_time[oi]),
                    "uav_arrival_time_h": float(arrival_time[oi]),
                    "delivery_time_h": float(delivery_time[oi]),
                    "lateness_h": float(lateness[oi]),
                    "demand": float(order.demand),
                }
            )

        return FusedModelSolution(
            selected_candidates=selected_candidates,
            order_plan=order_plan,
            drone_order_indices={int(k): list(v) for k, v in drone_order_indices.items()},
            rider_order_indices={int(k): list(v) for k, v in rider_order_indices.items()},
            objective_components=components,
            total_objective=total_objective,
            constraint_report=constraint_report,
        )


def solve_fused_delivery_model(
    calculator: UAVPathCostCalculator,
    merchants: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    candidate_points: List[Tuple[float, float]],
    n_drones: int = 3,
    n_riders: int = 4,
    order_count: Optional[int] = None,
    random_seed: Optional[int] = 42,
    config: Optional[FusionModelConfig] = None,
    render_plan: bool = False,
    render_save_path: Optional[str] = None,
    render_show: bool = False,
) -> FusedModelSolution:
    """融合模型的便捷入口函数。

    参数:
        render_plan: 是否在求解后渲染路径规划图
        render_save_path: 渲染图像输出路径（如 .png），为空则不落盘
        render_show: 是否弹出交互窗口显示图像
    """

    orders = build_orders_from_customers(
        customers=customers,
        merchants=merchants,
        order_count=order_count,
        random_seed=random_seed,
    )
    optimizer = DroneRiderFusionOptimizer(
        calculator=calculator,
        merchants=merchants,
        candidate_points=candidate_points,
        orders=orders,
        n_drones=n_drones,
        n_riders=n_riders,
        config=config,
    )
    solution = optimizer.solve()

    if render_plan or render_save_path:
        render_fused_solution_map(
            terrain=calculator.terrain,
            obstacles=calculator.obstacles,
            merchants=merchants,
            customers=customers,
            candidate_points=candidate_points,
            solution=solution,
            save_path=render_save_path,
            show=(render_show or render_save_path is None),
            safe_clearance_height=optimizer.config.safe_clearance_height,
            min_flight_height=optimizer.config.min_flight_height,
            max_flight_height=optimizer.config.max_flight_height,
        )

    return solution


if __name__ == "__main__":
    main()
