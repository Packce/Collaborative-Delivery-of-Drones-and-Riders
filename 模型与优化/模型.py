"""无人机路径代价模型。

该模块包含：
1. 地形插值模型（Terrain）
2. 建筑物障碍物模型（Obstacle）
3. 路径总代价计算器（UAVPathCostCalculator）
4. 从 CSV 自动加载地形与建筑数据的工具函数
"""

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, pi, sqrt
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MAX_DRONE_AGL_M = 120.0


def _bounded_flight_height_limits(
    min_flight_height: float,
    max_flight_height: float,
    hard_cap: float = MAX_DRONE_AGL_M,
) -> Tuple[float, float]:
    """返回合法的离地高度上下界（米）。"""

    lower = max(0.0, float(min_flight_height))
    upper = min(float(max_flight_height), float(hard_cap))
    if upper < lower:
        lower = upper
    return float(lower), float(upper)


def _resolve_cruise_agl(
    safe_clearance_height: float,
    min_flight_height: float,
    max_flight_height: float,
) -> float:
    """基于配置计算巡航离地高度（米），并强制不超过 120m。"""

    lower, upper = _bounded_flight_height_limits(min_flight_height, max_flight_height)
    target = max(lower, float(safe_clearance_height))
    return float(np.clip(target, lower, upper))


def _point_with_agl(
    terrain: "Terrain",
    x: float,
    y: float,
    agl_height: float,
) -> Tuple[float, float, float]:
    """按“离地高度”构造三维点。"""

    gx = float(x)
    gy = float(y)
    ground = terrain.get_elevation(gx, gy)
    return gx, gy, float(ground + float(agl_height))


def _normalize_algorithm_name(algorithm_name: Optional[str]) -> str:
    """将算法名规范化为可用于文件名的标签。"""

    raw = str(algorithm_name or "").strip().lower()
    if not raw:
        return "unknown"

    chars: List[str] = []
    for ch in raw:
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", ".", "/", "\\"}:
            chars.append("_")
        else:
            chars.append("_")

    tag = "".join(chars)
    while "__" in tag:
        tag = tag.replace("__", "_")
    tag = tag.strip("_")
    return tag or "unknown"


def _append_algorithm_suffix_to_stem(stem: str, algorithm_name: Optional[str]) -> str:
    """为文件 stem 追加算法后缀，避免重复追加。"""

    algo_tag = _normalize_algorithm_name(algorithm_name)
    suffix = f"_{algo_tag}"
    if stem.lower().endswith(suffix.lower()):
        return stem
    return f"{stem}{suffix}"


def _append_algorithm_suffix_to_path(path: Path, algorithm_name: Optional[str]) -> Path:
    """为文件路径追加算法后缀（位于扩展名前）。"""

    if path.suffix:
        new_stem = _append_algorithm_suffix_to_stem(path.stem, algorithm_name)
        new_name = f"{new_stem}{path.suffix}"
    else:
        new_name = _append_algorithm_suffix_to_stem(path.name, algorithm_name)
    return path.with_name(new_name)


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
            - 若离地高度 H-DH(x,y) > 120，超高部分按 c_T 惩罚。
            - 若 0 <= H-DH(x,y) <= 120，代价为 0。
            - 若 H < DH(x,y)，低于地形部分按 c_T 惩罚。

        参数:
            path: 路径点列表，每个点为 (x, y, H)

        返回:
            地形约束总成本。
        """

        total = 0.0
        for x, y, H in path:
            dh = self.terrain.get_elevation(x, y)
            agl = float(H - dh)
            if agl > MAX_DRONE_AGL_M:
                cost = (agl - MAX_DRONE_AGL_M) * self.c_T
            elif agl >= 0.0:
                cost = 0.0
            else:
                cost = (-agl) * self.c_T
            total += cost
        return float(total)

    def obstacle_collision_cost(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """计算障碍物相关代价（碰撞惩罚 + 高度奖励）。"""

        total_B, _ = self.obstacle_collision_detail(path, start, end)
        return float(total_B)

    def obstacle_collision_detail(
        self,
        path: List[Tuple[float, float, float]],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> Tuple[float, int]:
        """计算障碍物相关代价（碰撞惩罚 + 高度奖励）。

        参数:
            path: 中间路径点列表 (x, y, H)
            start: 起点 (x, y, H)
            end: 终点 (x, y, H)

        返回:
            (障碍物总代价, 有效避障次数)。

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
        collision_count = 0
        for idx, collide in enumerate(collision_flags):
            if collide and idx not in ignored_collision_indices:
                total_B += self.c_B
                collision_count += 1
            total_B += height_rewards[idx]

        return float(total_B), int(collision_count)

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
    project_root = Path(__file__).resolve().parents[1]
    merchant_csv = project_root / "数据" / "商家数据.csv"
    customer_csv = project_root / "数据" / "顾客数据.csv"
    NSGA_OPTION = False #改进的NSGA-II算法使用开关
    CLASSIC_GA_OPTION = False #经典遗传算法使用开关
    CLASSIC_NSGA_OPTION = True #传统NSGA算法使用开关
    PSO_OPTION = False #粒子群算法使用开关

    if (
        terrain_csv.exists()
        and building_csv.exists()
        and candidate_csv.exists()
        and merchant_csv.exists()
        and customer_csv.exists()
    ):
        terrain = load_terrain_from_csv(str(terrain_csv))
        obstacles = load_obstacles_from_csv(str(building_csv))
        candidate_points = load_candidates_from_csv(str(candidate_csv))
        merchants = select_random_merchants(str(merchant_csv), n=6, random_seed=42)
        customers = select_random_customers(str(customer_csv), m=120, random_seed=42)
        calculator = UAVPathCostCalculator(
            terrain=terrain,
            obstacles=obstacles,
            merchants=merchants,
            customers=customers,
            candidate_points=candidate_points,
            rider_speed=15.0,
            drone_speed=50.0,
        )

        print("Loaded real dataset successfully.")
        print(f"Terrain grid: {terrain.ny} x {terrain.nx}")
        print(f"Obstacle count: {len(obstacles)}")
        print(f"Candidate points count: {len(candidate_points)}")
        print(f"Sampled merchants: {len(merchants)}")
        print(f"Sampled customers: {len(customers)}")
        print(f"X range: [{terrain.x_min:.3f}, {terrain.x_max:.3f}]")
        print(f"Y range: [{terrain.y_min:.3f}, {terrain.y_max:.3f}]")

        if NSGA_OPTION:
            # 改进NSGA-II：改进的NSGA-II遗传算法
            print("Starting NSGA-II optimization...")
            cfg = FusionModelConfig(
                solver_mode="nsga2",
                ga_population_size=60,
                ga_generations=120,
                ga_verbose=True,
                max_selected_candidates=6,
                ga_random_seed=42,
            )
            output_dir = project_root / "统一输出"
            plan_path = output_dir / "最终优化路径图.png"

            solution = solve_fused_delivery_model(
                calculator=calculator,
                merchants=merchants,
                customers=customers,
                candidate_points=candidate_points,
                n_drones=8, # 无人机数量
                n_riders=20, # 骑手数量
                order_count=120,
                random_seed=42,
                config=cfg,
                render_plan=["uav", "rider"],
                render_save_path=str(plan_path),
                render_show=True,
                render_performance=True,
                performance_save_dir=str(output_dir),
                performance_show=False,
                performance_file_prefix="nsga2_compare",
            )
        elif CLASSIC_GA_OPTION:
            # 经典遗传算法
            print("Starting Classic Genetic Algorithm optimization...")
            cfg = FusionModelConfig(
                solver_mode="classic_ga",
                ga_population_size=60,
                ga_generations=120,
                ga_verbose=True,
                max_selected_candidates=6,
                ga_random_seed=42,
            )
            output_dir = project_root / "统一输出"
            plan_path = output_dir / "最终优化路径图.png"

            solution = solve_fused_delivery_model(
                calculator=calculator,
                merchants=merchants,
                customers=customers,
                candidate_points=candidate_points,
                n_drones=8, # 无人机数量
                n_riders=20, # 骑手数量
                order_count=120,
                random_seed=42,
                config=cfg,
                # render_plan=["uav", "rider"],
                # render_save_path=str(plan_path),
                # render_show=True,
                # render_performance=True,
                performance_save_dir=str(output_dir),
                performance_show=False,
                performance_file_prefix="classic_ga_compare",
            )
        elif CLASSIC_NSGA_OPTION:
            # 传统NSGA算法
            print("Starting Classic NSGA optimization...")
            cfg = FusionModelConfig(
                solver_mode="classic_nsga",
                ga_population_size=60,
                ga_generations=120,
                ga_verbose=True,
                max_selected_candidates=6,
                ga_random_seed=42,
            )
            output_dir = project_root / "统一输出"
            plan_path = output_dir / "最终优化路径图.png"

            solution = solve_fused_delivery_model(
                calculator=calculator,
                merchants=merchants,
                customers=customers,
                candidate_points=candidate_points,
                n_drones=8, # 无人机数量
                n_riders=20, # 骑手数量
                order_count=120,
                random_seed=42,
                config=cfg,
                # render_plan=["uav", "rider"],
                # render_save_path=str(plan_path),
                # render_show=True,
                # render_performance=True,
                performance_save_dir=str(output_dir),
                performance_show=False,
                performance_file_prefix="classic_nsga_compare",
            )
        elif PSO_OPTION:
            # 粒子群算法
            print("Starting Particle Swarm Optimization...")
            cfg = FusionModelConfig(
                solver_mode="pso",
                ga_population_size=60,
                ga_generations=120,
                ga_verbose=True,
                max_selected_candidates=6,
                ga_random_seed=42,
            )
            output_dir = project_root / "统一输出"
            plan_path = output_dir / "最终优化路径图.png"

            solution = solve_fused_delivery_model(
                calculator=calculator,
                merchants=merchants,
                customers=customers,
                candidate_points=candidate_points,
                n_drones=8, # 无人机数量
                n_riders=20, # 骑手数量
                order_count=120,
                random_seed=42,
                config=cfg,
                # render_plan=["uav", "rider"],
                # render_save_path=str(plan_path),
                # render_show=True,
                # render_performance=True,
                performance_save_dir=str(output_dir),
                performance_show=False,
                performance_file_prefix="pso_compare",
            )

        print()
        print(f"Solver mode: {solution.solver_mode}")
        print(f"Objective vector (f1,f2,f3): {solution.objective_vector}")
        print(f"Pareto size: {len(solution.pareto_front or [])}")
        print(f"Constraints OK: {solution.constraint_report.get('all_constraints_ok')}")
        print(f"Selected candidates: {solution.selected_candidates}")
        if solution.rendered_plan_paths:
            print("Rendered path maps:")
            for mode, path in solution.rendered_plan_paths.items():
                print(f"  - {mode}: {path}")
        if solution.performance_plot_paths:
            print("Rendered performance plots:")
            for name, path in solution.performance_plot_paths.items():
                print(f"  - {name}: {path}")
        if solution.csv_output_paths:
            print("Exported CSV reports:")
            for name, path in solution.csv_output_paths.items():
                print(f"  - {name}: {path}")
                 
    else:
        print("Default real-data files are incomplete; fallback to demo data.")
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
    solver_mode: str = "nsga2"  # "heuristic" or "nsga2"
    # solver_mode: str ="heuristic"
    ga_population_size: int = 60
    ga_generations: int = 120
    ga_crossover_prob: float = 0.8
    ga_mutation_prob: float = 0.2
    ga_layer_mutation_prob: float = 0.33
    ga_random_seed: Optional[int] = 42
    ga_force_mutation_if_none: bool = True
    ga_verbose: bool = False
    enable_multi_waypoint_search: bool = True
    path_grid_step_m: float = 120.0
    path_search_margin_m: float = 300.0
    path_obstacle_buffer_m: float = 20.0
    path_max_expand_nodes: int = 4000
    path_max_waypoints: int = 8
    enable_near_order_rider_only: bool = True
    near_order_rider_only_distance_m: float = 1000.0
    rider_speed_cooperative_kmh: float = 30.0
    rider_speed_rider_only_kmh: float = 12.0
    enforce_local_candidate_assignment: bool = True
    local_candidate_distance_ratio: float = 2.0
    local_candidate_extra_distance_m: float = 1200.0
    local_candidate_penalty_coeff: float = 200.0

    def __post_init__(self) -> None:
        min_h, max_h = _bounded_flight_height_limits(
            self.min_flight_height,
            self.max_flight_height,
        )
        self.min_flight_height = float(min_h)
        self.max_flight_height = float(max_h)
        self.safe_clearance_height = float(np.clip(self.safe_clearance_height, min_h, max_h))

        self.local_candidate_distance_ratio = max(1.0, float(self.local_candidate_distance_ratio))
        self.local_candidate_extra_distance_m = max(0.0, float(self.local_candidate_extra_distance_m))
        self.local_candidate_penalty_coeff = max(0.0, float(self.local_candidate_penalty_coeff))


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
    solver_mode: str = "heuristic"
    objective_vector: Optional[Tuple[float, float, float]] = None
    pareto_front: Optional[List[Dict[str, Any]]] = None
    performance_report: Optional[Dict[str, Any]] = None
    rendered_plan_paths: Optional[Dict[str, str]] = None
    performance_plot_paths: Optional[Dict[str, str]] = None
    csv_output_paths: Optional[Dict[str, str]] = None


@dataclass
class NSGA2Individual:
    """多目标遗传算法个体（三层编码）。"""

    order_sequence: List[int]          # O: 订单服务顺序（排列）
    candidate_assignment: List[int]    # D: 订单->候选点分配
    candidate_open: List[int]          # F: 候选点启用标记（0/1）
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    cv: float = float("inf")           # constraint violation
    rank: int = 10**9
    crowding_distance: float = 0.0
    decoded: Optional[Dict[str, Any]] = None


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
    obstacles: List[Obstacle],
    merchants: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    candidate_points: List[Tuple[float, float]],
    solution: FusedModelSolution,
    safe_clearance_height: float,
    min_flight_height: float,
    max_flight_height: float,
    rider_ground_lift: float,
    enable_multi_waypoint_search: bool = True,
    path_grid_step_m: float = 120.0,
    path_search_margin_m: float = 300.0,
    path_obstacle_buffer_m: float = 20.0,
    path_max_expand_nodes: int = 4000,
    path_max_waypoints: int = 8,
) -> List[Dict[str, Any]]:
    """将最终规划结果转换为可绘制的无人机/骑手路径。"""

    routes: List[Dict[str, Any]] = []
    if not solution.order_plan:
        return routes

    lift = max(0.5, float(rider_ground_lift))
    search_calculator = UAVPathCostCalculator(
        terrain=terrain,
        obstacles=obstacles,
    )
    # 渲染阶段按订单遍历时，同一商家->候选点航段可能重复出现，这里做缓存避免重复搜索。
    uav_path_cache: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    for plan in solution.order_plan:
        merchant_idx = int(plan.get("merchant_idx", -1))
        customer_idx = int(plan.get("customer_idx", -1))
        candidate_idx = int(plan.get("candidate_idx", -1))
        drone_idx = int(plan.get("drone_idx", -1))
        rider_idx = int(plan.get("rider_idx", -1))

        if not (0 <= merchant_idx < len(merchants)):
            continue

        merchant = merchants[merchant_idx]
        mx = float(merchant.get("x", 0.0))
        my = float(merchant.get("y", 0.0))

        if 0 <= customer_idx < len(customers):
            customer = customers[customer_idx]
            cx = float(customer.get("x", mx))
            cy = float(customer.get("y", my))
        else:
            cx, cy = mx, my

        mz = terrain.get_elevation(mx, my)
        cz = terrain.get_elevation(cx, cy)
        uav_path: List[Tuple[float, float, float]] = []

        if 0 <= candidate_idx < len(candidate_points) and drone_idx >= 0:
            gx, gy = candidate_points[candidate_idx]
            gx = float(gx)
            gy = float(gy)
            gz = terrain.get_elevation(gx, gy)

            cache_key = (merchant_idx, candidate_idx)
            uav_path = uav_path_cache.get(cache_key)
            if uav_path is None:
                cruise_agl = _resolve_cruise_agl(
                    safe_clearance_height=float(safe_clearance_height),
                    min_flight_height=float(min_flight_height),
                    max_flight_height=float(max_flight_height),
                )
                cruise_ref = max(mz, gz) + cruise_agl

                start = (mx, my, mz)
                end = _point_with_agl(terrain, gx, gy, cruise_agl)
                inner_points_raw = _search_uav_path_points(
                    calculator=search_calculator,
                    start=start,
                    end=end,
                    cruise=cruise_ref,
                    enable_multi_waypoint_search=bool(enable_multi_waypoint_search),
                    grid_step_m=float(path_grid_step_m),
                    search_margin_m=float(path_search_margin_m),
                    obstacle_buffer_m=float(path_obstacle_buffer_m),
                    max_expand_nodes=int(path_max_expand_nodes),
                    max_waypoints=int(path_max_waypoints),
                )
                inner_points = [
                    _point_with_agl(terrain, px, py, cruise_agl)
                    for px, py, _ in inner_points_raw
                ]
                uav_path = [start] + inner_points + [end]
                uav_path_cache[cache_key] = uav_path
            rider_path = [(gx, gy, gz + lift), (cx, cy, cz + lift)]
        else:
            rider_path = [(mx, my, mz + lift), (cx, cy, cz + lift)]

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
    enable_multi_waypoint_search: bool = True,
    path_grid_step_m: float = 120.0,
    path_search_margin_m: float = 300.0,
    path_obstacle_buffer_m: float = 20.0,
    path_max_expand_nodes: int = 4000,
    path_max_waypoints: int = 8,
    show_uav_paths: bool = True,
    show_rider_paths: bool = True,
    title: str = "Fused Delivery Plan",
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
    if not show_uav_paths and not show_rider_paths:
        raise ValueError("show_uav_paths 和 show_rider_paths 不能同时为 False。")

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
        obstacles=obstacles,
        merchants=merchants,
        customers=customers,
        candidate_points=candidate_points,
        solution=solution,
        safe_clearance_height=safe_clearance_height,
        min_flight_height=min_flight_height,
        max_flight_height=max_flight_height,
        rider_ground_lift=rider_ground_lift,
        enable_multi_waypoint_search=enable_multi_waypoint_search,
        path_grid_step_m=path_grid_step_m,
        path_search_margin_m=path_search_margin_m,
        path_obstacle_buffer_m=path_obstacle_buffer_m,
        path_max_expand_nodes=path_max_expand_nodes,
        path_max_waypoints=path_max_waypoints,
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
        drone_idx = int(route["drone_idx"])
        rider_idx = int(route["rider_idx"])

        drone_color = drone_palette[drone_idx % len(drone_palette)] if drone_idx >= 0 else drone_palette[0]
        rider_color = rider_palette[rider_idx % len(rider_palette)] if rider_idx >= 0 else rider_palette[0]

        if show_uav_paths and drone_idx >= 0:
            uav_path = np.asarray(route["uav_path"], dtype=float)
            if uav_path.ndim != 2 or uav_path.shape[0] < 2:
                continue
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

        if show_rider_paths:
            rider_path = np.asarray(route["rider_path"], dtype=float)
            if rider_path.ndim != 2 or rider_path.shape[0] < 2:
                continue
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

    plotter.add_text(title, position="upper_left", font_size=12, color=(0.12, 0.18, 0.26))

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
        # 纯离屏导出时，直接截图，避免 show() 在某些环境下阻塞。
        if saved_file:
            plotter.screenshot(saved_file)
        plotter.close()

    return saved_file


def _normalize_render_plan_modes(
    render_plan: Union[bool, str, Sequence[str], None],
) -> List[str]:
    """将 render_plan 参数归一化为渲染模式列表。"""

    if render_plan is None or render_plan is False:
        return []
    if render_plan is True:
        return ["fused"]

    if isinstance(render_plan, str):
        raw_items: List[str] = []
        for token in (
            render_plan.replace("|", ",")
            .replace(";", ",")
            .replace("+", ",")
            .replace("/", ",")
            .split(",")
        ):
            token = token.strip()
            if token:
                raw_items.append(token)
    elif isinstance(render_plan, Sequence):
        raw_items = [str(x).strip() for x in render_plan if str(x).strip()]
    else:
        raise TypeError("render_plan must be bool/str/Sequence[str]/None")

    alias_map: Dict[str, List[str]] = {
        "fused": ["fused"],
        "combined": ["fused"],
        "plan": ["fused"],
        "uav": ["uav"],
        "drone": ["uav"],
        "rider": ["rider"],
        "courier": ["rider"],
        "both": ["uav", "rider"],
        "split": ["uav", "rider"],
        "dual": ["uav", "rider"],
        "all": ["fused", "uav", "rider"],
        "full": ["fused", "uav", "rider"],
        "none": [],
        "融合": ["fused"],
        "总图": ["fused"],
        "无人机": ["uav"],
        "骑手": ["rider"],
        "双图": ["uav", "rider"],
        "分图": ["uav", "rider"],
        "全部": ["fused", "uav", "rider"],
    }

    modes: List[str] = []
    for item in raw_items:
        key = item.lower()
        if key not in alias_map:
            raise ValueError(
                f"Unsupported render_plan token: {item}. "
                "Supported: fused/uav/rider/both/all"
            )
        modes.extend(alias_map[key])

    unique_modes: List[str] = []
    seen = set()
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            unique_modes.append(mode)
    return unique_modes


def _resolve_render_output_paths(
    render_save_path: Optional[str],
    modes: List[str],
    algorithm_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """为每种渲染模式生成输出文件路径。"""

    if not modes:
        return {}

    if not render_save_path:
        return {mode: None for mode in modes}

    target = Path(render_save_path).expanduser()
    if len(modes) == 1:
        mode = modes[0]
        if target.suffix:
            single = _append_algorithm_suffix_to_path(target.resolve(), algorithm_name)
            single.parent.mkdir(parents=True, exist_ok=True)
            return {mode: str(single)}

        out_dir = target.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        single = (out_dir / f"fusion_plan_{mode}.png").resolve()
        single = _append_algorithm_suffix_to_path(single, algorithm_name)
        return {mode: str(single)}

    if target.suffix:
        parent = target.parent.resolve()
        stem = target.stem
        ext = target.suffix
        parent.mkdir(parents=True, exist_ok=True)
        result: Dict[str, Optional[str]] = {}
        for mode in modes:
            file_path = (parent / f"{stem}_{mode}{ext}").resolve()
            file_path = _append_algorithm_suffix_to_path(file_path, algorithm_name)
            result[mode] = str(file_path)
        return result

    out_dir = target.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Optional[str]] = {}
    for mode in modes:
        file_path = (out_dir / f"fusion_plan_{mode}.png").resolve()
        file_path = _append_algorithm_suffix_to_path(file_path, algorithm_name)
        result[mode] = str(file_path)
    return result


def _render_mode_to_flags(mode: str) -> Tuple[bool, bool, str]:
    """将渲染模式映射到路径图层开关。"""

    if mode == "uav":
        return True, False, "UAV Delivery Plan"
    if mode == "rider":
        return False, True, "Rider Delivery Plan"
    return True, True, "Fused Delivery Plan"


def render_fused_performance_plots(
    solution: FusedModelSolution,
    save_dir: Optional[str] = None,
    show: bool = False,
    file_prefix: str = "fusion_model",
    algorithm_name: Optional[str] = None,
) -> Dict[str, str]:
    """绘制融合模型性能图，便于与其他算法对比。"""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "render_fused_performance_plots 依赖 matplotlib，请先安装: pip install matplotlib"
        ) from exc

    output_dir: Optional[Path] = None
    if save_dir:
        output_dir = Path(save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not show:
        output_dir = (Path.cwd() / "统一输出").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    prefix_base = str(file_prefix).strip().replace(" ", "_") or "fusion_model"
    prefix = _append_algorithm_suffix_to_stem(prefix_base, algorithm_name)
    saved_paths: Dict[str, str] = {}

    def _finalize_figure(fig: Any, key: str, filename: str) -> None:
        if output_dir is not None:
            file_path = output_dir / filename
            fig.savefig(file_path, dpi=180, bbox_inches="tight")
            saved_paths[key] = str(file_path)
        if show:
            fig.show()
        plt.close(fig)

    report = solution.performance_report or {}
    trace = report.get("generation_trace", [])
    if isinstance(trace, list) and trace:
        gens = np.array([int(row.get("generation", i + 1)) for i, row in enumerate(trace)], dtype=float)
        best_total = np.array([float(row.get("best_weighted_total", np.nan)) for row in trace], dtype=float)
        mean_total = np.array([float(row.get("mean_weighted_total", np.nan)) for row in trace], dtype=float)
        feasible_ratio = np.array([float(row.get("feasible_ratio", np.nan)) for row in trace], dtype=float)
        front0_size = np.array([float(row.get("front0_size", np.nan)) for row in trace], dtype=float)
        best_f1 = np.array([float(row.get("best_f1", np.nan)) for row in trace], dtype=float)
        best_f2 = np.array([float(row.get("best_f2", np.nan)) for row in trace], dtype=float)
        best_f3 = np.array([float(row.get("best_f3", np.nan)) for row in trace], dtype=float)

        fig1, ax1 = plt.subplots(figsize=(9.5, 5.4))
        ax1.plot(gens, best_total, color="#d1495b", linewidth=2.0, label="Best Weighted Objective")
        ax1.plot(gens, mean_total, color="#2a9d8f", linewidth=1.8, label="Mean Weighted Objective")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Weighted Objective")
        ax1.set_title("Optimization Convergence")
        ax1.grid(alpha=0.28)
        ax1.legend()
        _finalize_figure(fig1, "convergence", f"{prefix}_convergence.png")

        fig2, ax2 = plt.subplots(figsize=(9.5, 5.4))
        ax2.plot(gens, best_f1, color="#ef476f", linewidth=2.0, label="f1: cost+makespan")
        ax2.plot(gens, best_f2, color="#118ab2", linewidth=2.0, label="f2: late+open")
        ax2.plot(gens, best_f3, color="#06d6a0", linewidth=2.0, label="f3: drone_cost")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Objective Value")
        ax2.set_title("Objective Evolution")
        ax2.grid(alpha=0.28)
        ax2.legend()
        _finalize_figure(fig2, "objectives", f"{prefix}_objectives.png")

        fig3, ax3 = plt.subplots(figsize=(9.5, 5.4))
        ax3.plot(gens, feasible_ratio, color="#5e60ce", linewidth=2.0, label="Feasible Ratio")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Feasible Ratio")
        ax3.set_ylim(0.0, 1.05)
        ax3.grid(alpha=0.28)
        ax3_t = ax3.twinx()
        ax3_t.plot(gens, front0_size, color="#f4a261", linewidth=1.8, linestyle="--", label="Front-0 Size")
        ax3_t.set_ylabel("Front-0 Size")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_t.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax3.set_title("Feasibility And Pareto Pressure")
        _finalize_figure(fig3, "feasibility", f"{prefix}_feasibility.png")

    pareto = solution.pareto_front or []
    pareto_obj: List[Tuple[float, float, float]] = []
    for item in pareto:
        vec = item.get("objective_vector")
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            pareto_obj.append((float(vec[0]), float(vec[1]), float(vec[2])))

    if pareto_obj:
        arr = np.asarray(pareto_obj, dtype=float)
        fig4, ax4 = plt.subplots(figsize=(8.8, 6.0))
        sc = ax4.scatter(
            arr[:, 0],
            arr[:, 1],
            c=arr[:, 2],
            cmap="viridis",
            s=34,
            alpha=0.88,
            edgecolors="none",
        )
        ax4.set_xlabel("f1: cost+makespan")
        ax4.set_ylabel("f2: late+open")
        ax4.set_title("Final Pareto Front (color=f3)")
        ax4.grid(alpha=0.25)
        cbar = fig4.colorbar(sc, ax=ax4)
        cbar.set_label("f3: drone_cost")
        _finalize_figure(fig4, "pareto", f"{prefix}_pareto_front.png")

    components = solution.objective_components or {}
    if components:
        labels = list(components.keys())
        values = np.array([float(components[k]) for k in labels], dtype=float)
        fig5, ax5 = plt.subplots(figsize=(10.0, 5.6))
        bars = ax5.bar(
            np.arange(len(labels)),
            values,
            color=["#457b9d", "#f4a261", "#2a9d8f", "#e76f51", "#6d597a"][: len(labels)],
        )
        ax5.set_xticks(np.arange(len(labels)))
        ax5.set_xticklabels(labels, rotation=24, ha="right")
        ax5.set_ylabel("Value")
        ax5.set_title("Final Objective Components")
        ax5.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, values):
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        _finalize_figure(fig5, "components", f"{prefix}_objective_components.png")

    return saved_paths


def _resolve_tabular_output_dir(
    csv_output_dir: Optional[str],
    render_save_path: Optional[str],
    performance_save_dir: Optional[str],
) -> Path:
    """解析表格输出目录。"""

    if csv_output_dir:
        out_dir = Path(csv_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    if performance_save_dir:
        out_dir = Path(performance_save_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    if render_save_path:
        target = Path(render_save_path).expanduser()
        out_dir = target.resolve().parent if target.suffix else target.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    out_dir = (Path.cwd() / "统一输出").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _entity_display_name(entity: Dict[str, Any], idx: int, prefix: str) -> str:
    """生成实体显示名，优先 name/id。"""

    name = str(entity.get("name", "")).strip()
    if name:
        return name
    eid = str(entity.get("id", "")).strip()
    if eid:
        return eid
    return f"{prefix}{idx}"


def _segment_intersects_any_obstacle(
    p1_xy: Tuple[float, float],
    p2_xy: Tuple[float, float],
    obstacles_xy: List[Tuple[float, float, float]],
) -> bool:
    """判断二维线段是否与任一障碍物（含缓冲半径）相交。"""

    x1, y1 = float(p1_xy[0]), float(p1_xy[1])
    x2, y2 = float(p2_xy[0]), float(p2_xy[1])
    min_x, max_x = (x1, x2) if x1 <= x2 else (x2, x1)
    min_y, max_y = (y1, y2) if y1 <= y2 else (y2, y1)

    for ox, oy, rr in obstacles_xy:
        if ox < min_x - rr or ox > max_x + rr or oy < min_y - rr or oy > max_y + rr:
            continue
        if UAVPathCostCalculator._segment_intersect_cylinder(
            (x1, y1, 0.0),
            (x2, y2, 0.0),
            float(ox),
            float(oy),
            float(rr),
        ):
            return True
    return False


def _nearest_free_grid_index(
    valid_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    px: float,
    py: float,
) -> Optional[Tuple[int, int]]:
    """返回距离给定点最近的可行网格索引。"""

    h, w = valid_mask.shape
    best: Optional[Tuple[int, int]] = None
    best_d2 = float("inf")
    for iy in range(h):
        yv = float(ys[iy])
        for ix in range(w):
            if not bool(valid_mask[iy, ix]):
                continue
            xv = float(xs[ix])
            d2 = (xv - px) ** 2 + (yv - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (ix, iy)
    return best


def _visibility_simplify_polyline(
    points_xy: List[Tuple[float, float]],
    obstacles_xy: List[Tuple[float, float, float]],
) -> List[Tuple[float, float]]:
    """按可视性裁剪折线，减少不必要拐点。"""

    if len(points_xy) <= 2:
        return points_xy

    simplified: List[Tuple[float, float]] = [points_xy[0]]
    i = 0
    n = len(points_xy)
    while i < n - 1:
        next_idx = i + 1
        for j in range(n - 1, i, -1):
            if not _segment_intersects_any_obstacle(points_xy[i], points_xy[j], obstacles_xy):
                next_idx = j
                break
        simplified.append(points_xy[next_idx])
        i = next_idx
    return simplified


def _search_uav_path_points(
    calculator: UAVPathCostCalculator,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    cruise: float,
    enable_multi_waypoint_search: bool,
    grid_step_m: float,
    search_margin_m: float,
    obstacle_buffer_m: float,
    max_expand_nodes: int,
    max_waypoints: int,
) -> List[Tuple[float, float, float]]:
    """从起点到终点搜索多拐点路径；失败时回退为单中点。"""

    sx, sy, _ = start
    gx, gy, _ = end
    mid_fallback = ((sx + gx) * 0.5, (sy + gy) * 0.5, float(cruise))

    if not enable_multi_waypoint_search or not calculator.obstacles:
        return [mid_fallback]

    step = max(20.0, float(grid_step_m))
    buffer_m = max(0.0, float(obstacle_buffer_m))
    max_nodes = max(200, int(max_expand_nodes))
    max_wp = max(1, int(max_waypoints))

    terrain = calculator.terrain
    direct_dist = sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
    margin = max(float(search_margin_m), step * 2.0, direct_dist * 0.25)

    min_x = max(float(terrain.x_min), min(sx, gx) - margin)
    max_x = min(float(terrain.x_max), max(sx, gx) + margin)
    min_y = max(float(terrain.y_min), min(sy, gy) - margin)
    max_y = min(float(terrain.y_max), max(sy, gy) + margin)

    if max_x - min_x < step or max_y - min_y < step:
        return [mid_fallback]

    xs = np.arange(min_x, max_x + 0.5 * step, step, dtype=float)
    ys = np.arange(min_y, max_y + 0.5 * step, step, dtype=float)
    if xs.size < 2 or ys.size < 2:
        return [mid_fallback]

    obstacles_xy: List[Tuple[float, float, float]] = []
    for obs in calculator.obstacles:
        rr = float(obs.r) + buffer_m
        ox = float(obs.x)
        oy = float(obs.y)
        if ox + rr < min_x or ox - rr > max_x or oy + rr < min_y or oy - rr > max_y:
            continue
        obstacles_xy.append((ox, oy, rr))

    if not obstacles_xy:
        return [mid_fallback]

    if not _segment_intersects_any_obstacle((sx, sy), (gx, gy), obstacles_xy):
        return [mid_fallback]

    valid = np.ones((ys.size, xs.size), dtype=bool)
    for iy in range(ys.size):
        yv = float(ys[iy])
        for ix in range(xs.size):
            xv = float(xs[ix])
            for ox, oy, rr in obstacles_xy:
                if (xv - ox) ** 2 + (yv - oy) ** 2 <= rr ** 2:
                    valid[iy, ix] = False
                    break

    start_idx = _nearest_free_grid_index(valid, xs, ys, float(sx), float(sy))
    goal_idx = _nearest_free_grid_index(valid, xs, ys, float(gx), float(gy))
    if start_idx is None or goal_idx is None:
        return [mid_fallback]

    if start_idx == goal_idx:
        return [mid_fallback]

    moves = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    edge_block_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}
    g_score: Dict[Tuple[int, int], float] = {start_idx: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        ax, ay = float(xs[a[0]]), float(ys[a[1]])
        bx, by = float(xs[b[0]]), float(ys[b[1]])
        return float(sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (_heuristic(start_idx, goal_idx), 0.0, start_idx))
    closed: set = set()
    expanded = 0

    while open_heap and expanded < max_nodes:
        _, curr_g, curr = heapq.heappop(open_heap)
        if curr in closed:
            continue
        if curr == goal_idx:
            break
        closed.add(curr)
        expanded += 1

        cx, cy = curr
        for dx, dy in moves:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= xs.size or ny < 0 or ny >= ys.size:
                continue
            if not bool(valid[ny, nx]):
                continue

            nxt = (nx, ny)
            edge_key = (curr, nxt) if curr <= nxt else (nxt, curr)
            blocked = edge_block_cache.get(edge_key)
            if blocked is None:
                p1 = (float(xs[cx]), float(ys[cy]))
                p2 = (float(xs[nx]), float(ys[ny]))
                blocked = _segment_intersects_any_obstacle(p1, p2, obstacles_xy)
                edge_block_cache[edge_key] = blocked
            if blocked:
                continue

            step_cost = sqrt((float(xs[nx]) - float(xs[cx])) ** 2 + (float(ys[ny]) - float(ys[cy])) ** 2)
            tentative_g = float(curr_g + step_cost)
            prev_best = g_score.get(nxt, float("inf"))
            if tentative_g + 1e-9 >= prev_best:
                continue

            came_from[nxt] = curr
            g_score[nxt] = tentative_g
            f_score = tentative_g + _heuristic(nxt, goal_idx)
            heapq.heappush(open_heap, (f_score, tentative_g, nxt))

    if goal_idx not in came_from:
        return [mid_fallback]

    path_idx: List[Tuple[int, int]] = [goal_idx]
    cur = goal_idx
    while cur != start_idx:
        cur = came_from.get(cur)
        if cur is None:
            return [mid_fallback]
        path_idx.append(cur)
    path_idx.reverse()

    grid_points_xy = [(float(xs[ix]), float(ys[iy])) for ix, iy in path_idx]
    polyline_xy = [(float(sx), float(sy))] + grid_points_xy + [(float(gx), float(gy))]
    simplified_xy = _visibility_simplify_polyline(polyline_xy, obstacles_xy)
    inner_xy = simplified_xy[1:-1]

    if not inner_xy:
        return [mid_fallback]

    if len(inner_xy) > max_wp:
        select_idx = np.linspace(0, len(inner_xy) - 1, max_wp, dtype=int)
        inner_xy = [inner_xy[int(i)] for i in select_idx.tolist()]

    return [(float(x), float(y), float(cruise)) for x, y in inner_xy]


def _build_uav_leg_report(
    calculator: UAVPathCostCalculator,
    merchant: Dict[str, Any],
    candidate_point: Tuple[float, float],
    safe_clearance_height: float,
    min_flight_height: float,
    max_flight_height: float,
    enable_multi_waypoint_search: bool = True,
    path_grid_step_m: float = 120.0,
    path_search_margin_m: float = 300.0,
    path_obstacle_buffer_m: float = 20.0,
    path_max_expand_nodes: int = 4000,
    path_max_waypoints: int = 8,
    drone_cruise_power_w: float = 420.0,
    drone_battery_energy_wh: float = 588.0,
) -> Dict[str, float]:
    """构建单条商家->候选点航段的明细指标。"""

    mx = float(merchant.get("x", 0.0))
    my = float(merchant.get("y", 0.0))
    gx = float(candidate_point[0])
    gy = float(candidate_point[1])

    terrain = calculator.terrain
    mz = terrain.get_elevation(mx, my)
    gz = terrain.get_elevation(gx, gy)
    cruise_agl = _resolve_cruise_agl(
        safe_clearance_height=float(safe_clearance_height),
        min_flight_height=float(min_flight_height),
        max_flight_height=float(max_flight_height),
    )
    cruise_ref = max(mz, gz) + cruise_agl

    start = (mx, my, mz)
    end = _point_with_agl(terrain, gx, gy, cruise_agl)
    path_raw = _search_uav_path_points(
        calculator=calculator,
        start=start,
        end=end,
        cruise=cruise_ref,
        enable_multi_waypoint_search=bool(enable_multi_waypoint_search),
        grid_step_m=float(path_grid_step_m),
        search_margin_m=float(path_search_margin_m),
        obstacle_buffer_m=float(path_obstacle_buffer_m),
        max_expand_nodes=int(path_max_expand_nodes),
        max_waypoints=int(path_max_waypoints),
    )
    path = [
        _point_with_agl(terrain, px, py, cruise_agl)
        for px, py, _ in path_raw
    ]

    terrain_cost_value = float(calculator.terrain_cost(path))
    obstacle_cost_value, collision_count = calculator.obstacle_collision_detail(path, start, end)
    flight_distance_m = float(calculator.flight_distance_cost(path, start, end))
    altitude_penalty = float(calculator.altitude_variation_cost(path, start, end))
    turning_penalty = float(calculator.turning_angle_cost(path, start, end))
    total_cost = float(
        terrain_cost_value
        + obstacle_cost_value
        + flight_distance_m
        + altitude_penalty
        + turning_penalty
    )

    straight_distance_m = float(sqrt((gx - mx) ** 2 + (gy - my) ** 2))
    speed_km_h = max(float(calculator.drone_speed), 1e-9)
    flight_time_h = float((flight_distance_m / 1000.0) / speed_km_h)
    flight_time_min = float(flight_time_h * 60.0)

    battery_wh = max(float(drone_battery_energy_wh), 1e-9)
    energy_percent = float((float(drone_cruise_power_w) * flight_time_h / battery_wh) * 100.0)

    return {
        "straight_distance_m": straight_distance_m,
        "actual_distance_m": flight_distance_m,
        "flight_time_min": flight_time_min,
        "energy_percent": energy_percent,
        "collision_count": int(collision_count),
        "path_cost": total_cost,
        "altitude_penalty": altitude_penalty,
        "turning_penalty": turning_penalty,
    }


def _write_csv_rows(
    file_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    """按 UTF-8-SIG 写入 CSV。"""

    with file_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_fused_solution_csv_reports(
    calculator: UAVPathCostCalculator,
    merchants: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    candidate_points: List[Tuple[float, float]],
    orders: List[DeliveryOrder],
    solution: FusedModelSolution,
    output_dir: Path,
    rider_count_for_rider_only: Optional[int] = None,
    rider_capacity_for_rider_only: float = float("inf"),
    rider_speed_for_rider_only_kmh: float = 15.0,
    safe_clearance_height: float = 20.0,
    min_flight_height: float = 20.0,
    max_flight_height: float = 120.0,
    enable_multi_waypoint_search: bool = True,
    path_grid_step_m: float = 120.0,
    path_search_margin_m: float = 300.0,
    path_obstacle_buffer_m: float = 20.0,
    path_max_expand_nodes: int = 4000,
    path_max_waypoints: int = 8,
    algorithm_name: Optional[str] = None,
) -> Dict[str, str]:
    """导出融合模型的三类 CSV 报表。"""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_tag = _normalize_algorithm_name(algorithm_name)

    import time
    timestamp = time.strftime("_%Y%m%d_%H%M%S")

    selected_candidates: List[int] = []
    seen_candidates = set()
    for g in solution.selected_candidates:
        gi = int(g)
        if 0 <= gi < len(candidate_points) and gi not in seen_candidates:
            seen_candidates.add(gi)
            selected_candidates.append(gi)

    selected_rows: List[Dict[str, Any]] = []
    for idx, gi in enumerate(selected_candidates, start=1):
        gx, gy = candidate_points[gi]
        gz = calculator.terrain.get_elevation(gx, gy)
        selected_rows.append(
            {
                "编号": int(idx),
                "候选点索引": int(gi),
                "X": round(float(gx), 6),
                "Y": round(float(gy), 6),
                "Z": round(float(gz), 6),
            }
        )

    selected_path = output_dir / f"最终所选起降点列表_{algo_tag}{timestamp}.csv"
    _write_csv_rows(
        selected_path,
        fieldnames=["编号", "候选点索引", "X", "Y", "Z"],
        rows=selected_rows,
    )

    leg_cache: Dict[Tuple[int, int], Dict[str, float]] = {}

    def _cached_leg(mi: int, gi: int) -> Dict[str, float]:
        key = (int(mi), int(gi))
        if key not in leg_cache:
            leg_cache[key] = _build_uav_leg_report(
                calculator=calculator,
                merchant=merchants[mi],
                candidate_point=candidate_points[gi],
                safe_clearance_height=safe_clearance_height,
                min_flight_height=min_flight_height,
                max_flight_height=max_flight_height,
                enable_multi_waypoint_search=enable_multi_waypoint_search,
                path_grid_step_m=path_grid_step_m,
                path_search_margin_m=path_search_margin_m,
                path_obstacle_buffer_m=path_obstacle_buffer_m,
                path_max_expand_nodes=path_max_expand_nodes,
                path_max_waypoints=path_max_waypoints,
            )
        return leg_cache[key]

    route_rows: List[Dict[str, Any]] = []
    for mi, merchant in enumerate(merchants):
        start_name = _entity_display_name(merchant, mi, "M")
        for gi in selected_candidates:
            leg = _cached_leg(mi, gi)
            route_rows.append(
                {
                    "起点": start_name,
                    "终点": f"G{gi}",
                    "直线距离_m": round(float(leg["straight_distance_m"]), 4),
                    "实际距离_m": round(float(leg["actual_distance_m"]), 4),
                    "飞行时间_分钟": round(float(leg["flight_time_min"]), 4),
                    "耗电量_百分比": round(float(leg["energy_percent"]), 4),
                    "避障次数": int(leg["collision_count"]),
                    "路径代价": round(float(leg["path_cost"]), 4),
                    "高度惩罚": round(float(leg["altitude_penalty"]), 4),
                    "转向惩罚": round(float(leg["turning_penalty"]), 4),
                }
            )

    route_path = output_dir / f"商家到所选起降点无人机运输路线结果_{algo_tag}{timestamp}.csv"
    _write_csv_rows(
        route_path,
        fieldnames=[
            "起点",
            "终点",
            "直线距离_m",
            "实际距离_m",
            "飞行时间_分钟",
            "耗电量_百分比",
            "避障次数",
            "路径代价",
            "高度惩罚",
            "转向惩罚",
        ],
        rows=route_rows,
    )

    def _simulate_rider_only_delivery_duration_h() -> List[float]:
        """估计“仅骑手”运筹下每单从发起到送达的时长（小时）。"""

        n_orders = len(orders)
        if n_orders <= 0:
            return []

        if rider_count_for_rider_only is not None:
            n_riders = max(1, int(rider_count_for_rider_only))
        else:
            used_riders = [int(k) for k in solution.rider_order_indices.keys()]
            inferred = (max(used_riders) + 1) if used_riders else 1
            n_riders = max(1, inferred)

        rider_speed = max(1e-9, float(rider_speed_for_rider_only_kmh))
        rider_capacity = float(rider_capacity_for_rider_only)

        finish_time_h = [float(order.earliest_time) for order in orders]
        rider_available_h = [0.0] * n_riders
        rider_pos_xy: List[Optional[Tuple[float, float]]] = [None] * n_riders

        sorted_orders = sorted(range(n_orders), key=lambda oi: float(orders[oi].earliest_time))
        for oi in sorted_orders:
            order = orders[oi]
            mi = int(order.merchant_idx)
            if 0 <= mi < len(merchants):
                mx = float(merchants[mi].get("x", 0.0))
                my = float(merchants[mi].get("y", 0.0))
            else:
                mx = float(order.customer_x)
                my = float(order.customer_y)

            best_r = 0
            best_finish = float("inf")
            feasible_found = False

            for r in range(n_riders):
                if order.demand > rider_capacity + 1e-9:
                    continue

                pos = rider_pos_xy[r]
                to_pickup_m = 0.0 if pos is None else sqrt((pos[0] - mx) ** 2 + (pos[1] - my) ** 2)
                to_pickup_h = (to_pickup_m / 1000.0) / rider_speed

                depart_h = max(rider_available_h[r] + to_pickup_h, float(order.earliest_time))
                leg_m = sqrt((mx - order.customer_x) ** 2 + (my - order.customer_y) ** 2)
                leg_h = (leg_m / 1000.0) / rider_speed
                finish_h = depart_h + leg_h + float(order.service_time)

                if finish_h < best_finish:
                    best_finish = finish_h
                    best_r = r
                    feasible_found = True

            if not feasible_found:
                best_r = int(np.argmin(rider_available_h))
                pos = rider_pos_xy[best_r]
                to_pickup_m = 0.0 if pos is None else sqrt((pos[0] - mx) ** 2 + (pos[1] - my) ** 2)
                to_pickup_h = (to_pickup_m / 1000.0) / rider_speed
                depart_h = max(rider_available_h[best_r] + to_pickup_h, float(order.earliest_time))
                leg_m = sqrt((mx - order.customer_x) ** 2 + (my - order.customer_y) ** 2)
                leg_h = (leg_m / 1000.0) / rider_speed
                best_finish = depart_h + leg_h + float(order.service_time)

            finish_time_h[oi] = best_finish
            rider_available_h[best_r] = best_finish
            rider_pos_xy[best_r] = (float(order.customer_x), float(order.customer_y))

        return [max(0.0, finish_time_h[oi] - float(orders[oi].earliest_time)) for oi in range(n_orders)]

    rider_only_duration_h = _simulate_rider_only_delivery_duration_h()

    order_rows: List[Dict[str, Any]] = []
    for plan in solution.order_plan:
        oi = int(plan.get("order_index", -1))
        if oi < 0 or oi >= len(orders):
            continue
        order = orders[oi]
        mi = int(plan.get("merchant_idx", order.merchant_idx))
        ci = int(plan.get("customer_idx", order.customer_idx))
        gi = int(plan.get("candidate_idx", -1))

        if 0 <= mi < len(merchants):
            start_name = _entity_display_name(merchants[mi], mi, "M")
        else:
            start_name = f"M{mi}"

        if 0 <= ci < len(customers):
            end_name = _entity_display_name(customers[ci], ci, "C")
        else:
            end_name = f"C{ci}"

        launch_h = float(order.earliest_time)
        arrive_h = float(plan.get("delivery_time_h", launch_h))
        uav_depart_h = float(plan.get("uav_depart_time_h", launch_h))
        uav_arrive_h = float(plan.get("uav_arrival_time_h", uav_depart_h))

        total_delivery_h = max(0.0, arrive_h - launch_h)
        uav_delivery_h = max(0.0, uav_arrive_h - uav_depart_h)
        rider_delivery_h = max(0.0, arrive_h - uav_arrive_h)
        rider_only_h = rider_only_duration_h[oi] if oi < len(rider_only_duration_h) else 0.0
        straight_distance_m = 0.0

        if 0 <= mi < len(merchants) and 0 <= ci < len(customers):
            mx = float(merchants[mi].get("x", 0.0))
            my = float(merchants[mi].get("y", 0.0))
            cx = float(customers[ci].get("x", 0.0))
            cy = float(customers[ci].get("y", 0.0))
            straight_distance_m = sqrt((mx - cx) ** 2 + (my - cy) ** 2)

        total_energy = 0.0
        if 0 <= mi < len(merchants) and 0 <= gi < len(candidate_points):
            total_energy = float(_cached_leg(mi, gi)["energy_percent"])

        order_rows.append(
            {
                "起点": start_name,
                "终点": end_name,
                "直线距离": round(straight_distance_m, 4),
                "订单发起时间": round(launch_h * 60.0, 4),
                "到达时间": round(arrive_h * 60.0, 4),
                "总配送时间": round(total_delivery_h * 60.0, 4),
                "无人机配送时间": round(uav_delivery_h * 60.0, 4),
                "骑手配送时间": round(rider_delivery_h * 60.0, 4),
                "仅骑手直送时间": round(rider_only_h * 60.0, 4),
                "总能耗": round(total_energy, 4),
            }
        )

    order_path = output_dir / f"模拟订单详细配送数据_{algo_tag}{timestamp}.csv"
    _write_csv_rows(
        order_path,
        fieldnames=[
            "起点",
            "终点",
            "直线距离",
            "订单发起时间",
            "到达时间",
            "总配送时间",
            "无人机配送时间",
            "骑手配送时间",
            "仅骑手直送时间",
            "总能耗",
        ],
        rows=order_rows,
    )

    return {
        "selected_candidates": str(selected_path),
        "merchant_to_candidate_routes": str(route_path),
        "simulated_orders": str(order_path),
    }


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
        self._rng = np.random.default_rng(self.config.ga_random_seed)

        self._uav_cost: Optional[np.ndarray] = None
        self._uav_time: Optional[np.ndarray] = None
        self._uav_energy: Optional[np.ndarray] = None
        self._order_candidate_score: Optional[np.ndarray] = None
        self._order_candidate_rider_distance: Optional[np.ndarray] = None
        self._order_local_distance_limit: Optional[np.ndarray] = None
        self.last_performance_report: Optional[Dict[str, Any]] = None
        self._merchant_customer_distance_m = self._build_merchant_customer_distance_cache()
        self._rider_only_flags = self._build_rider_only_flags()

    @staticmethod
    def _distance_2d(ax: float, ay: float, bx: float, by: float) -> float:
        return float(sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    @staticmethod
    def _hours_from_distance(distance_m: float, speed_km_h: float) -> float:
        speed = max(1e-9, speed_km_h)
        return float((distance_m / 1000.0) / speed)

    def _cooperative_rider_speed(self) -> float:
        """协同配送（无人机+骑手）场景下的骑手速度（km/h）。"""

        return max(1e-9, float(self.config.rider_speed_cooperative_kmh))

    def _build_merchant_customer_distance_cache(self) -> List[float]:
        distance_cache: List[float] = []
        for order in self.orders:
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0
            merchant = self.merchants[mi]
            mx = float(merchant.get("x", 0.0))
            my = float(merchant.get("y", 0.0))
            distance_cache.append(self._distance_2d(mx, my, order.customer_x, order.customer_y))
        return distance_cache

    def _build_rider_only_flags(self) -> List[bool]:
        if not bool(self.config.enable_near_order_rider_only):
            return [False] * len(self.orders)

        threshold_m = max(0.0, float(self.config.near_order_rider_only_distance_m))
        return [dist <= threshold_m + 1e-9 for dist in self._merchant_customer_distance_m]

    def _is_rider_only_order(self, order_idx: int) -> bool:
        if order_idx < 0 or order_idx >= len(self.orders):
            return False
        return bool(self._rider_only_flags[order_idx])

    def _drone_order_indices(self) -> List[int]:
        return [oi for oi in range(len(self.orders)) if not self._is_rider_only_order(oi)]

    def _pickup_xy_for_order(self, order_idx: int, candidate_idx: int) -> Tuple[float, float]:
        order = self.orders[order_idx]
        if self._is_rider_only_order(order_idx):
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0
            merchant = self.merchants[mi]
            return float(merchant.get("x", 0.0)), float(merchant.get("y", 0.0))

        if 0 <= candidate_idx < len(self.candidate_points):
            gx, gy = self.candidate_points[candidate_idx]
            return float(gx), float(gy)

        return float(order.customer_x), float(order.customer_y)

    def _normalize_order_sequence(self, sequence: List[int]) -> List[int]:
        """将任意序列修复为 0..n-1 的合法排列。"""

        n_orders = len(self.orders)
        seen = set()
        normalized: List[int] = []
        for x in sequence:
            try:
                oi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= oi < n_orders and oi not in seen:
                normalized.append(oi)
                seen.add(oi)
        for oi in range(n_orders):
            if oi not in seen:
                normalized.append(oi)
        return normalized

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
        cruise_agl = _resolve_cruise_agl(
            safe_clearance_height=self.config.safe_clearance_height,
            min_flight_height=self.config.min_flight_height,
            max_flight_height=self.config.max_flight_height,
        )
        cruise_ref = max(mz, gz) + cruise_agl

        start = (mx, my, mz)
        end = _point_with_agl(terrain, gx, gy, cruise_agl)
        path_raw = _search_uav_path_points(
            calculator=self.calculator,
            start=start,
            end=end,
            cruise=cruise_ref,
            enable_multi_waypoint_search=bool(self.config.enable_multi_waypoint_search),
            grid_step_m=float(self.config.path_grid_step_m),
            search_margin_m=float(self.config.path_search_margin_m),
            obstacle_buffer_m=float(self.config.path_obstacle_buffer_m),
            max_expand_nodes=int(self.config.path_max_expand_nodes),
            max_waypoints=int(self.config.path_max_waypoints),
        )
        path = [
            _point_with_agl(terrain, px, py, cruise_agl)
            for px, py, _ in path_raw
        ]

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
        rider_dist_matrix = np.zeros((no, ng), dtype=float)
        local_limit = np.zeros(no, dtype=float)

        ratio_limit = float(self.config.local_candidate_distance_ratio)
        extra_limit = float(self.config.local_candidate_extra_distance_m)
        locality_penalty_coeff = float(self.config.local_candidate_penalty_coeff)
        enforce_local = bool(self.config.enforce_local_candidate_assignment)

        for oi, order in enumerate(self.orders):
            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0

            for gi, (gx, gy) in enumerate(self.candidate_points):
                gx = float(gx)
                gy = float(gy)
                rider_dist = self._distance_2d(order.customer_x, order.customer_y, gx, gy)
                rider_dist_matrix[oi, gi] = rider_dist

            nearest_dist = float(np.min(rider_dist_matrix[oi, :])) if ng > 0 else 0.0
            local_limit_oi = max(nearest_dist * ratio_limit, nearest_dist + extra_limit)
            local_limit[oi] = float(local_limit_oi)

            for gi in range(ng):
                rider_dist = float(rider_dist_matrix[oi, gi])
                value = float(
                    self.config.lambda_drone_cost * self._uav_cost[mi, gi]
                    + self.config.lambda_rider_cost * rider_dist
                )
                if enforce_local and rider_dist > local_limit_oi + 1e-9:
                    overflow = rider_dist - local_limit_oi
                    value += locality_penalty_coeff * overflow
                score[oi, gi] = value

        self._order_candidate_score = score
        self._order_candidate_rider_distance = rider_dist_matrix
        self._order_local_distance_limit = local_limit

    def _is_local_candidate_for_order(self, order_idx: int, candidate_idx: int) -> bool:
        if not bool(self.config.enforce_local_candidate_assignment):
            return True

        if self._order_candidate_rider_distance is None or self._order_local_distance_limit is None:
            self._build_order_candidate_score_matrix()
        assert self._order_candidate_rider_distance is not None and self._order_local_distance_limit is not None

        if order_idx < 0 or order_idx >= len(self.orders):
            return False
        if candidate_idx < 0 or candidate_idx >= len(self.candidate_points):
            return False

        rider_dist = float(self._order_candidate_rider_distance[order_idx, candidate_idx])
        dist_limit = float(self._order_local_distance_limit[order_idx])
        return rider_dist <= dist_limit + 1e-9

    def _local_candidate_indices_for_order(self, order_idx: int) -> List[int]:
        if order_idx < 0 or order_idx >= len(self.orders):
            return []
        if not bool(self.config.enforce_local_candidate_assignment):
            return list(range(len(self.candidate_points)))

        if self._order_candidate_rider_distance is None or self._order_local_distance_limit is None:
            self._build_order_candidate_score_matrix()
        assert self._order_candidate_rider_distance is not None and self._order_local_distance_limit is not None

        dist_row = self._order_candidate_rider_distance[order_idx, :]
        limit = float(self._order_local_distance_limit[order_idx])
        return [
            int(gi)
            for gi in np.where(dist_row <= limit + 1e-9)[0].tolist()
        ]

    def _ensure_local_coverage_for_selected(
        self,
        selected_candidates: List[int],
        drone_orders: List[int],
        max_pick: int,
    ) -> List[int]:
        if not drone_orders:
            return []
        if not bool(self.config.enforce_local_candidate_assignment):
            return sorted({int(g) for g in selected_candidates if 0 <= int(g) < len(self.candidate_points)})

        if self._order_candidate_score is None:
            self._build_order_candidate_score_matrix()
        assert self._order_candidate_score is not None

        ng = len(self.candidate_points)
        selected_set = {int(g) for g in selected_candidates if 0 <= int(g) < ng}

        for oi in drone_orders:
            if any(self._is_local_candidate_for_order(oi, g) for g in selected_set):
                continue

            local_options = self._local_candidate_indices_for_order(oi)
            if not local_options:
                continue

            best_local = min(local_options, key=lambda g: float(self._order_candidate_score[oi, g]))
            if best_local in selected_set:
                continue

            if len(selected_set) < max_pick:
                selected_set.add(best_local)
                continue

            if not selected_set:
                selected_set.add(best_local)
                continue

            support = {g: 0 for g in selected_set}
            for oj in drone_orders:
                best_curr = min(support.keys(), key=lambda g: float(self._order_candidate_score[oj, g]))
                support[best_curr] += 1

            removable = [g for g, cnt in support.items() if cnt == 0]
            if not removable:
                continue

            remove_g = max(
                removable,
                key=lambda g: float(np.mean(self._order_candidate_score[drone_orders, g])),
            )
            selected_set.remove(remove_g)
            selected_set.add(best_local)

        return sorted(selected_set)

    def select_candidates_greedy(self) -> List[int]:
        """按总成本下降幅度进行候选点贪心选择。"""

        if self._order_candidate_score is None:
            self._build_order_candidate_score_matrix()

        assert self._order_candidate_score is not None
        drone_orders = self._drone_order_indices()
        if not drone_orders:
            return []

        active_score = self._order_candidate_score[drone_orders, :]
        no, ng = active_score.shape
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
                v = np.minimum(current_best, active_score[:, g])
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
            selected = [int(np.argmin(np.mean(active_score, axis=0)))]
        selected = self._ensure_local_coverage_for_selected(selected, drone_orders, max_pick)
        return selected

    def _assign_candidates(self, selected_candidates: List[int]) -> Tuple[List[int], Dict[int, float]]:
        if self._order_candidate_score is None:
            self._build_order_candidate_score_matrix()
        assert self._order_candidate_score is not None

        if not selected_candidates and self._drone_order_indices():
            best_default = int(np.argmin(np.mean(self._order_candidate_score[self._drone_order_indices(), :], axis=0)))
            selected_candidates = [best_default]

        candidate_load = {g: 0.0 for g in selected_candidates}
        assigned_candidate = [-1] * len(self.orders)

        for oi, order in enumerate(self.orders):
            if self._is_rider_only_order(oi):
                assigned_candidate[oi] = -1
                continue

            best_g = None
            best_score = float("inf")
            local_selected = [g for g in selected_candidates if self._is_local_candidate_for_order(oi, g)]
            candidate_pool = local_selected if local_selected else list(selected_candidates)
            fallback_g = (
                min(candidate_pool, key=lambda g: float(self._order_candidate_score[oi, g]))
                if candidate_pool
                else -1
            )

            for g in candidate_pool:
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
                chosen = fallback_g if fallback_g >= 0 else 0
            else:
                chosen = best_g

            assigned_candidate[oi] = chosen
            candidate_load.setdefault(chosen, 0.0)
            candidate_load[chosen] += float(order.demand)

        return assigned_candidate, candidate_load

    def _assign_drones(
        self,
        assigned_candidate: List[int],
        order_sequence: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float], List[float], Dict[int, List[int]], List[float]]:
        assert self._uav_cost is not None and self._uav_time is not None and self._uav_energy is not None

        n_orders = len(self.orders)
        assigned_drone = [-1] * n_orders
        depart_time = [0.0] * n_orders
        arrival_time = [0.0] * n_orders

        drone_available = [0.0] * self.n_drones
        drone_energy_used = [0.0] * self.n_drones
        drone_order_indices: Dict[int, List[int]] = defaultdict(list)

        if order_sequence is None:
            sorted_orders = sorted(range(n_orders), key=lambda oi: self.orders[oi].earliest_time)
        else:
            sorted_orders = self._normalize_order_sequence(order_sequence)
        for oi in sorted_orders:
            order = self.orders[oi]
            if self._is_rider_only_order(oi):
                ready_time = float(order.earliest_time)
                assigned_drone[oi] = -1
                depart_time[oi] = ready_time
                arrival_time[oi] = ready_time
                continue

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
        rider_speed = self._cooperative_rider_speed()
        t_ij = self._hours_from_distance(dist_ij, rider_speed) + order_i.service_time

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
        d_t_equiv = d_t_h * self._cooperative_rider_speed() * 1000.0
        alpha = float(np.clip(self.config.alpha_spatial, 0.0, 1.0))
        return float(alpha * d_ij + (1.0 - alpha) * d_t_equiv)

    def _rider_transition_cost(
        self,
        prev_order_idx: Optional[int],
        current_order_idx: int,
        pickup_x: float,
        pickup_y: float,
    ) -> float:
        """计算骑手从上一个服务节点转移到当前订单的时空代价。"""

        current_order = self.orders[current_order_idx]
        pickup_x = float(pickup_x)
        pickup_y = float(pickup_y)

        if prev_order_idx is None:
            spatial_m = self._distance_2d(pickup_x, pickup_y, current_order.customer_x, current_order.customer_y)
            temporal_equiv_m = 0.0
        else:
            prev_order = self.orders[prev_order_idx]
            reposition_m = self._distance_2d(prev_order.customer_x, prev_order.customer_y, pickup_x, pickup_y)
            delivery_m = self._distance_2d(pickup_x, pickup_y, current_order.customer_x, current_order.customer_y)
            spatial_m = reposition_m + delivery_m
            temporal_h = self._temporal_distance_component(prev_order, current_order)
            temporal_equiv_m = temporal_h * self._cooperative_rider_speed() * 1000.0

        alpha = float(np.clip(self.config.alpha_spatial, 0.0, 1.0))
        return float(alpha * spatial_m + (1.0 - alpha) * temporal_equiv_m)

    def _assign_riders(
        self,
        assigned_candidate: List[int],
        arrival_time: List[float],
        order_sequence: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float], List[float], Dict[int, List[int]], float]:
        n_orders = len(self.orders)
        assigned_rider = [-1] * n_orders
        delivery_time = [0.0] * n_orders
        lateness = [0.0] * n_orders
        rider_order_indices: Dict[int, List[int]] = defaultdict(list)

        rider_available = [0.0] * self.n_riders
        rider_pos: List[Optional[Tuple[float, float]]] = [None] * self.n_riders
        rider_spatiotemporal_cost = 0.0

        if order_sequence is None:
            sorted_orders = sorted(range(n_orders), key=lambda oi: arrival_time[oi])
        else:
            sorted_orders = self._normalize_order_sequence(order_sequence)
        rider_speed = self._cooperative_rider_speed()
        for oi in sorted_orders:
            order = self.orders[oi]
            g = assigned_candidate[oi]
            pickup_x, pickup_y = self._pickup_xy_for_order(oi, g)

            best_r = 0
            best_finish = float("inf")
            best_depart = 0.0
            feasible_found = False

            for r in range(self.n_riders):
                if order.demand > self.config.rider_capacity:
                    continue

                pos = rider_pos[r]
                to_pickup_m = 0.0 if pos is None else self._distance_2d(pos[0], pos[1], pickup_x, pickup_y)
                to_pickup_h = self._hours_from_distance(to_pickup_m, rider_speed)
                depart = max(rider_available[r] + to_pickup_h, arrival_time[oi], order.earliest_time)

                leg_m = self._distance_2d(pickup_x, pickup_y, order.customer_x, order.customer_y)
                leg_h = self._hours_from_distance(leg_m, rider_speed)
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
                to_pickup_m = 0.0 if pos is None else self._distance_2d(pos[0], pos[1], pickup_x, pickup_y)
                to_pickup_h = self._hours_from_distance(to_pickup_m, rider_speed)
                best_depart = max(rider_available[best_r] + to_pickup_h, arrival_time[oi], order.earliest_time)
                leg_m = self._distance_2d(pickup_x, pickup_y, order.customer_x, order.customer_y)
                leg_h = self._hours_from_distance(leg_m, rider_speed)
                best_finish = best_depart + leg_h + order.service_time

            assigned_rider[oi] = best_r
            delivery_time[oi] = best_finish
            lateness[oi] = max(0.0, best_finish - order.latest_time)

            previous_orders = rider_order_indices[best_r]
            prev_idx = previous_orders[-1] if previous_orders else None
            rider_spatiotemporal_cost += self._rider_transition_cost(prev_idx, oi, pickup_x, pickup_y)

            rider_order_indices[best_r].append(oi)
            rider_available[best_r] = best_finish
            rider_pos[best_r] = (order.customer_x, order.customer_y)

        return assigned_rider, delivery_time, lateness, rider_order_indices, float(rider_spatiotemporal_cost)

    def _compute_cost_components(
        self,
        selected_candidates: List[int],
        assigned_candidate: List[int],
        assigned_drone: List[int],
        delivery_time: List[float],
        lateness: List[float],
        rider_spatiotemporal_cost: float,
    ) -> Dict[str, float]:
        """计算统一的成本分量，供单目标和多目标复用。"""

        assert self._uav_cost is not None and self._uav_energy is not None

        drone_cost = 0.0
        drone_energy = 0.0
        for oi, order in enumerate(self.orders):
            if self._is_rider_only_order(oi):
                continue

            mi = int(order.merchant_idx)
            if mi < 0 or mi >= len(self.merchants):
                mi = 0
            g = int(assigned_candidate[oi])
            if 0 <= g < len(self.candidate_points):
                drone_cost += float(self._uav_cost[mi, g])
                drone_energy += float(self._uav_energy[mi, g])
            else:
                drone_cost += 1e6
                drone_energy += 1e6

        late_cost = self.config.late_penalty_coeff * float(sum(lateness))
        open_cost = self.config.candidate_opening_cost * float(len(selected_candidates))
        makespan = max(delivery_time) if delivery_time else 0.0

        return {
            "drone_cost": float(drone_cost),
            "rider_spatiotemporal_cost": float(rider_spatiotemporal_cost),
            "late_cost": float(late_cost),
            "open_cost": float(open_cost),
            "makespan": float(makespan),
            "drone_energy": float(drone_energy),
            "n_selected_candidates": float(len(selected_candidates)),
            "n_assigned_drones": float(len({int(d) for d in assigned_drone if int(d) >= 0})),
        }

    def _weighted_total_from_components(self, components: Dict[str, float]) -> float:
        """按原有加权方式计算单值总目标（用于兼容输出与打分）。"""

        return float(
            self.config.lambda_drone_cost * components["drone_cost"]
            + self.config.lambda_rider_cost * components["rider_spatiotemporal_cost"]
            + self.config.lambda_late_cost * components["late_cost"]
            + self.config.lambda_open_cost * components["open_cost"]
            + self.config.lambda_makespan * components["makespan"]
        )

    def _evaluate_objective(
        self,
        selected_candidates: List[int],
        assigned_candidate: List[int],
        assigned_drone: List[int],
        delivery_time: List[float],
        lateness: List[float],
        rider_spatiotemporal_cost: float,
    ) -> Tuple[Dict[str, float], float]:
        components = self._compute_cost_components(
            selected_candidates=selected_candidates,
            assigned_candidate=assigned_candidate,
            assigned_drone=assigned_drone,
            delivery_time=delivery_time,
            lateness=lateness,
            rider_spatiotemporal_cost=rider_spatiotemporal_cost,
        )
        weighted_total = self._weighted_total_from_components(components)
        return components, weighted_total

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
        selected_set = set(int(g) for g in selected_candidates)
        report["candidate_limit_ok"] = len(selected_candidates) <= self.config.max_selected_candidates
        report["all_orders_have_candidate"] = True
        report["all_orders_have_drone"] = True
        report["all_orders_have_rider"] = all(r >= 0 for r in assigned_rider)
        report["candidate_activation_ok"] = True
        report["local_candidate_assignment_ok"] = True
        report["rider_only_rule_ok"] = True
        report["candidate_capacity_ok"] = all(
            load <= self.config.candidate_capacity + 1e-9 for load in candidate_load.values()
        )

        drone_capacity_ok = True
        for oi, order in enumerate(self.orders):
            g = int(assigned_candidate[oi])
            d = int(assigned_drone[oi])
            rider_only = self._is_rider_only_order(oi)
            if rider_only:
                if g != -1 or d != -1:
                    report["all_orders_have_candidate"] = False
                    report["all_orders_have_drone"] = False
                    report["rider_only_rule_ok"] = False
                continue

            if g < 0:
                report["all_orders_have_candidate"] = False
            if d < 0:
                report["all_orders_have_drone"] = False
            if g not in selected_set:
                report["candidate_activation_ok"] = False
            if g >= 0 and not self._is_local_candidate_for_order(oi, g):
                report["local_candidate_assignment_ok"] = False
            if order.demand > self.config.drone_capacity + 1e-9:
                drone_capacity_ok = False
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

    def _build_order_plan(
        self,
        assigned_candidate: List[int],
        assigned_drone: List[int],
        assigned_rider: List[int],
        depart_time: List[float],
        arrival_time: List[float],
        delivery_time: List[float],
        lateness: List[float],
    ) -> List[Dict[str, Any]]:
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
                    "delivery_mode": "rider_only" if self._is_rider_only_order(oi) else "uav_rider",
                    "uav_depart_time_h": float(depart_time[oi]),
                    "uav_arrival_time_h": float(arrival_time[oi]),
                    "delivery_time_h": float(delivery_time[oi]),
                    "lateness_h": float(lateness[oi]),
                    "demand": float(order.demand),
                }
            )
        return order_plan

    def _candidate_load_from_assignment(self, assigned_candidate: List[int]) -> Dict[int, float]:
        candidate_load: Dict[int, float] = defaultdict(float)
        for oi, g in enumerate(assigned_candidate):
            if 0 <= g < len(self.candidate_points):
                candidate_load[int(g)] += float(self.orders[oi].demand)
        return dict(candidate_load)

    def _constraint_violation_value(
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
    ) -> float:
        """连续化约束违反度（CV）。CV=0 表示可行。"""

        cv = 0.0
        selected_set = set(selected_candidates)
        max_selected = max(1, int(self.config.max_selected_candidates))

        cv += max(0.0, float(len(selected_candidates) - max_selected))

        if np.isfinite(self.config.candidate_capacity):
            for load in candidate_load.values():
                cv += max(0.0, float(load - self.config.candidate_capacity))

        for oi, order in enumerate(self.orders):
            rider_only = self._is_rider_only_order(oi)
            g = int(assigned_candidate[oi])
            d = int(assigned_drone[oi])
            if rider_only:
                if g != -1:
                    cv += 1.0
                if d != -1:
                    cv += 1.0
            else:
                if g < 0 or g >= len(self.candidate_points):
                    cv += 1.0
                elif g not in selected_set:
                    cv += 1.0
                elif not self._is_local_candidate_for_order(oi, g):
                    cv += 1.0
                if d < 0:
                    cv += 1.0
                cv += max(0.0, float(order.demand - self.config.drone_capacity))
            cv += max(0.0, float(order.demand - self.config.rider_capacity))

        if np.isfinite(self.config.drone_energy_limit):
            for e in drone_energy_used:
                cv += max(0.0, float(e - self.config.drone_energy_limit))

        for r in assigned_rider:
            if r < 0:
                cv += 1.0

        for idxs in drone_order_indices.values():
            if not idxs:
                continue
            idxs_sorted = sorted(idxs, key=lambda i: depart_time[i])
            for pos in range(1, len(idxs_sorted)):
                prev_idx = idxs_sorted[pos - 1]
                curr_idx = idxs_sorted[pos]
                required_gap = arrival_time[prev_idx] + self.config.drone_turnaround_time
                cv += max(0.0, float(required_gap - depart_time[curr_idx]))

        return float(cv)

    def _repair_chromosome(
        self, individual: NSGA2Individual
    ) -> Tuple[List[int], List[int], List[int], List[int], Dict[int, float]]:
        """修复三层染色体，保证基本可行性。"""

        assert self._order_candidate_score is not None
        n_orders = len(self.orders)
        n_candidates = len(self.candidate_points)
        drone_orders = self._drone_order_indices()
        drone_order_set = set(drone_orders)
        has_drone_orders = bool(drone_orders)
        max_selected = max(1, min(int(self.config.max_selected_candidates), n_candidates)) if has_drone_orders else 0

        order_sequence = self._normalize_order_sequence(individual.order_sequence)

        assignment: List[int] = []
        for g in individual.candidate_assignment:
            try:
                gi = int(g)
            except (TypeError, ValueError):
                gi = 0
            gi = int(np.clip(gi, 0, n_candidates - 1))
            assignment.append(gi)
        if len(assignment) < n_orders:
            assignment.extend([0] * (n_orders - len(assignment)))
        assignment = assignment[:n_orders]

        open_bits: List[int] = []
        for v in individual.candidate_open:
            try:
                open_bits.append(1 if int(v) != 0 else 0)
            except (TypeError, ValueError):
                open_bits.append(0)
        if len(open_bits) < n_candidates:
            open_bits.extend([0] * (n_candidates - len(open_bits)))
        open_bits = open_bits[:n_candidates]

        selected = [g for g, bit in enumerate(open_bits) if bit == 1]
        if has_drone_orders and not selected:
            active_mean = np.mean(self._order_candidate_score[drone_orders, :], axis=0)
            selected = [int(np.argmin(active_mean))]
        if not has_drone_orders:
            selected = []

        if len(selected) > max_selected:
            # 优先保留当前承载订单多且评分更优的候选点
            counts = {g: 0 for g in selected}
            active_mean = np.mean(self._order_candidate_score[drone_orders, :], axis=0)
            for oi, g in enumerate(assignment):
                if oi in drone_order_set and g in counts:
                    counts[g] += 1
            selected = sorted(
                selected,
                key=lambda g: (-counts[g], float(active_mean[g])),
            )[:max_selected]

        if has_drone_orders:
            selected = self._ensure_local_coverage_for_selected(selected, drone_orders, max_selected)
        selected_set = set(selected)

        for oi in range(n_orders):
            if self._is_rider_only_order(oi):
                assignment[oi] = -1
                continue
            if not selected_set:
                best_g = int(np.argmin(self._order_candidate_score[oi, :]))
                selected_set.add(best_g)
                selected.append(best_g)
            local_selected = [g for g in selected_set if self._is_local_candidate_for_order(oi, g)]
            candidate_pool = local_selected if local_selected else list(selected_set)
            if assignment[oi] not in candidate_pool:
                best_g = min(candidate_pool, key=lambda g: float(self._order_candidate_score[oi, g]))
                assignment[oi] = int(best_g)

        candidate_load = self._candidate_load_from_assignment(assignment)

        if np.isfinite(self.config.candidate_capacity) and selected_set:
            cap = float(self.config.candidate_capacity)
            overloaded = [g for g, load in candidate_load.items() if load > cap + 1e-9]
            for g in overloaded:
                if g not in selected_set:
                    continue
                order_ids = [oi for oi in range(n_orders) if assignment[oi] == g and not self._is_rider_only_order(oi)]
                order_ids.sort(key=lambda oi: float(self.orders[oi].demand), reverse=True)
                for oi in order_ids:
                    if candidate_load.get(g, 0.0) <= cap + 1e-9:
                        break
                    demand = float(self.orders[oi].demand)
                    feasible_targets = [
                        t for t in selected
                        if t != g and candidate_load.get(t, 0.0) + demand <= cap + 1e-9
                    ]
                    local_feasible_targets = [t for t in feasible_targets if self._is_local_candidate_for_order(oi, t)]
                    if local_feasible_targets:
                        feasible_targets = local_feasible_targets
                    if not feasible_targets:
                        continue
                    best_t = min(feasible_targets, key=lambda t: float(self._order_candidate_score[oi, t]))
                    assignment[oi] = int(best_t)
                    candidate_load[g] = candidate_load.get(g, 0.0) - demand
                    candidate_load[best_t] = candidate_load.get(best_t, 0.0) + demand

        open_bits = [1 if g in selected_set else 0 for g in range(n_candidates)]
        selected = sorted(selected_set)
        candidate_load = self._candidate_load_from_assignment(assignment)
        return order_sequence, assignment, open_bits, selected, candidate_load

    def _evaluate_nsga2_individual(self, individual: NSGA2Individual) -> NSGA2Individual:
        """评估单个 NSGA-II 个体。"""

        assert self._uav_cost is not None and self._uav_time is not None and self._uav_energy is not None
        assert self._order_candidate_score is not None

        try:
            order_sequence, assignment, open_bits, selected, candidate_load = self._repair_chromosome(individual)
            individual.order_sequence = order_sequence
            individual.candidate_assignment = assignment
            individual.candidate_open = open_bits

            assigned_drone, depart_time, arrival_time, drone_order_indices, drone_energy_used = self._assign_drones(
                assignment, order_sequence=order_sequence
            )
            assigned_rider, delivery_time, lateness, rider_order_indices, rider_st_cost = self._assign_riders(
                assignment, arrival_time, order_sequence=order_sequence
            )

            components = self._compute_cost_components(
                selected_candidates=selected,
                assigned_candidate=assignment,
                assigned_drone=assigned_drone,
                delivery_time=delivery_time,
                lateness=lateness,
                rider_spatiotemporal_cost=rider_st_cost,
            )
            weighted_total = self._weighted_total_from_components(components)

            # 三目标：运输成本/效率、时窗惩罚+开站成本、能耗
            f1 = float(
                components["drone_cost"]
                + components["rider_spatiotemporal_cost"]
                + self.config.lambda_makespan * components["makespan"]
            )
            f2 = float(components["late_cost"] + components["open_cost"])
            f3 = float(components["drone_energy"])

            constraint_report = self._build_constraint_report(
                selected_candidates=selected,
                assigned_candidate=assignment,
                assigned_drone=assigned_drone,
                assigned_rider=assigned_rider,
                drone_order_indices=drone_order_indices,
                drone_energy_used=drone_energy_used,
                depart_time=depart_time,
                arrival_time=arrival_time,
                candidate_load=candidate_load,
            )
            cv = self._constraint_violation_value(
                selected_candidates=selected,
                assigned_candidate=assignment,
                assigned_drone=assigned_drone,
                assigned_rider=assigned_rider,
                drone_order_indices=drone_order_indices,
                drone_energy_used=drone_energy_used,
                depart_time=depart_time,
                arrival_time=arrival_time,
                candidate_load=candidate_load,
            )

            individual.objectives = (f1, f2, f3)
            individual.cv = float(cv)
            individual.decoded = {
                "selected_candidates": selected,
                "assigned_candidate": assignment,
                "assigned_drone": assigned_drone,
                "assigned_rider": assigned_rider,
                "depart_time": depart_time,
                "arrival_time": arrival_time,
                "delivery_time": delivery_time,
                "lateness": lateness,
                "drone_order_indices": drone_order_indices,
                "rider_order_indices": rider_order_indices,
                "drone_energy_used": drone_energy_used,
                "candidate_load": candidate_load,
                "constraint_report": constraint_report,
                "components": components,
                "weighted_total": weighted_total,
            }
        except InfeasibleFusionPlanError:
            individual.objectives = (1e12, 1e12, 1e12)
            individual.cv = 1e9
            individual.decoded = None

        return individual

    def _random_individual(self) -> NSGA2Individual:
        n_orders = len(self.orders)
        n_candidates = len(self.candidate_points)
        drone_orders = self._drone_order_indices()
        max_selected = max(1, min(int(self.config.max_selected_candidates), n_candidates)) if drone_orders else 0
        if max_selected > 0:
            n_open = int(self._rng.integers(1, max_selected + 1))
            selected = self._rng.choice(n_candidates, size=n_open, replace=False).tolist()
        else:
            selected = []

        open_bits = [0] * n_candidates
        for g in selected:
            open_bits[int(g)] = 1

        assignment: List[int] = []
        for oi in range(n_orders):
            if self._is_rider_only_order(oi):
                assignment.append(-1)
            elif selected:
                assignment.append(int(self._rng.choice(selected)))
            else:
                assignment.append(0)
        order_sequence = self._rng.permutation(n_orders).tolist()

        return NSGA2Individual(
            order_sequence=order_sequence,
            candidate_assignment=assignment,
            candidate_open=open_bits,
        )

    @staticmethod
    def _clone_individual(individual: NSGA2Individual) -> NSGA2Individual:
        return NSGA2Individual(
            order_sequence=list(individual.order_sequence),
            candidate_assignment=list(individual.candidate_assignment),
            candidate_open=list(individual.candidate_open),
            objectives=tuple(individual.objectives),
            cv=float(individual.cv),
            rank=int(individual.rank),
            crowding_distance=float(individual.crowding_distance),
            decoded=individual.decoded,
        )

    @staticmethod
    def _is_feasible(individual: NSGA2Individual) -> bool:
        return individual.cv <= 1e-9

    def _constrained_dominates(self, a: NSGA2Individual, b: NSGA2Individual) -> bool:
        a_feasible = self._is_feasible(a)
        b_feasible = self._is_feasible(b)
        if a_feasible and not b_feasible:
            return True
        if not a_feasible and b_feasible:
            return False
        if not a_feasible and not b_feasible:
            return a.cv < b.cv - 1e-12

        better_or_equal = True
        strictly_better = False
        for k in range(3):
            if a.objectives[k] > b.objectives[k] + 1e-12:
                better_or_equal = False
                break
            if a.objectives[k] < b.objectives[k] - 1e-12:
                strictly_better = True
        return better_or_equal and strictly_better

    def _fast_nondominated_sort(self, population: List[NSGA2Individual]) -> List[List[int]]:
        n = len(population)
        dominates: List[List[int]] = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts: List[List[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._constrained_dominates(population[p], population[q]):
                    dominates[p].append(q)
                elif self._constrained_dominates(population[q], population[p]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                population[p].rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[int] = []
            for p in fronts[i]:
                for q in dominates[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        population[q].rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _assign_crowding_distance(self, population: List[NSGA2Individual], front: List[int]) -> None:
        if not front:
            return
        for idx in front:
            population[idx].crowding_distance = 0.0
        if len(front) <= 2:
            for idx in front:
                population[idx].crowding_distance = float("inf")
            return

        n_obj = 3
        for m in range(n_obj):
            sorted_idx = sorted(front, key=lambda i: population[i].objectives[m])
            population[sorted_idx[0]].crowding_distance = float("inf")
            population[sorted_idx[-1]].crowding_distance = float("inf")

            f_min = population[sorted_idx[0]].objectives[m]
            f_max = population[sorted_idx[-1]].objectives[m]
            denom = f_max - f_min
            if abs(denom) <= 1e-12:
                continue

            for pos in range(1, len(sorted_idx) - 1):
                i_prev = sorted_idx[pos - 1]
                i_curr = sorted_idx[pos]
                i_next = sorted_idx[pos + 1]
                distance = (
                    population[i_next].objectives[m] - population[i_prev].objectives[m]
                ) / denom
                population[i_curr].crowding_distance += float(distance)

    def _tournament_pick(self, population: List[NSGA2Individual]) -> NSGA2Individual:
        if len(population) == 1:
            return population[0]

        i, j = self._rng.choice(len(population), size=2, replace=False)
        a = population[int(i)]
        b = population[int(j)]

        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        if a.crowding_distance > b.crowding_distance:
            return a
        if b.crowding_distance > a.crowding_distance:
            return b
        if a.cv < b.cv:
            return a
        if b.cv < a.cv:
            return b
        return a if self._rng.random() < 0.5 else b

    def _order_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        n = len(p1)
        if n < 2:
            return list(p1), list(p2)
        left = int(self._rng.integers(0, n - 1))
        right = int(self._rng.integers(left + 1, n + 1))

        def _ox(a: List[int], b: List[int]) -> List[int]:
            child = [-1] * n
            segment = a[left:right]
            child[left:right] = segment
            fill = [x for x in b if x not in segment]
            ptr = 0
            for pos in range(n):
                if child[pos] == -1:
                    child[pos] = fill[ptr]
                    ptr += 1
            return child

        return _ox(p1, p2), _ox(p2, p1)

    def _segment_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        n = len(p1)
        if n < 2:
            return list(p1), list(p2)
        cut = int(self._rng.integers(1, n))
        c1 = list(p1[:cut]) + list(p2[cut:])
        c2 = list(p2[:cut]) + list(p1[cut:])
        return c1, c2

    def _uniform_bit_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        c1: List[int] = []
        c2: List[int] = []
        for i in range(len(p1)):
            if self._rng.random() < 0.5:
                c1.append(int(p1[i]))
                c2.append(int(p2[i]))
            else:
                c1.append(int(p2[i]))
                c2.append(int(p1[i]))
        return c1, c2

    def _crossover(self, p1: NSGA2Individual, p2: NSGA2Individual) -> Tuple[NSGA2Individual, NSGA2Individual]:
        if self._rng.random() > float(np.clip(self.config.ga_crossover_prob, 0.0, 1.0)):
            return self._clone_individual(p1), self._clone_individual(p2)

        o1, o2 = self._order_crossover(p1.order_sequence, p2.order_sequence)
        d1, d2 = self._segment_crossover(p1.candidate_assignment, p2.candidate_assignment)
        f1, f2 = self._uniform_bit_crossover(p1.candidate_open, p2.candidate_open)

        c1 = NSGA2Individual(order_sequence=o1, candidate_assignment=d1, candidate_open=f1)
        c2 = NSGA2Individual(order_sequence=o2, candidate_assignment=d2, candidate_open=f2)
        return c1, c2

    def _mutate(self, individual: NSGA2Individual) -> None:
        if self._rng.random() > float(np.clip(self.config.ga_mutation_prob, 0.0, 1.0)):
            return

        n_orders = len(individual.order_sequence)
        n_candidates = len(individual.candidate_open)
        layer_prob = float(np.clip(self.config.ga_layer_mutation_prob, 0.0, 1.0))
        mutated = False

        if n_orders >= 2 and self._rng.random() < layer_prob:
            a, b = self._rng.choice(n_orders, size=2, replace=False)
            ia = int(a)
            ib = int(b)
            individual.order_sequence[ia], individual.order_sequence[ib] = (
                individual.order_sequence[ib],
                individual.order_sequence[ia],
            )
            mutated = True

        if n_orders >= 1 and n_candidates >= 1 and self._rng.random() < layer_prob:
            oi = int(self._rng.integers(0, n_orders))
            gi = int(self._rng.integers(0, n_candidates))
            individual.candidate_assignment[oi] = gi
            mutated = True

        if n_candidates >= 1 and self._rng.random() < layer_prob:
            gi = int(self._rng.integers(0, n_candidates))
            individual.candidate_open[gi] = 1 - int(individual.candidate_open[gi])
            mutated = True

        if not mutated and self.config.ga_force_mutation_if_none and n_candidates >= 1:
            gi = int(self._rng.integers(0, n_candidates))
            individual.candidate_open[gi] = 1 - int(individual.candidate_open[gi])

    def _initialize_population(self) -> List[NSGA2Individual]:
        pop_size = max(4, int(self.config.ga_population_size))
        population: List[NSGA2Individual] = []
        for _ in range(pop_size):
            ind = self._random_individual()
            self._evaluate_nsga2_individual(ind)
            population.append(ind)
        return population

    def _environmental_selection(
        self, merged_population: List[NSGA2Individual], target_size: int
    ) -> List[NSGA2Individual]:
        fronts = self._fast_nondominated_sort(merged_population)
        new_pop: List[NSGA2Individual] = []

        for front in fronts:
            self._assign_crowding_distance(merged_population, front)
            if len(new_pop) + len(front) <= target_size:
                new_pop.extend(merged_population[idx] for idx in front)
            else:
                sorted_front = sorted(
                    front,
                    key=lambda idx: merged_population[idx].crowding_distance,
                    reverse=True,
                )
                remain = target_size - len(new_pop)
                new_pop.extend(merged_population[idx] for idx in sorted_front[:remain])
                break

        return new_pop

    def _collect_generation_metrics(
        self,
        population: List[NSGA2Individual],
        generation: int,
        elapsed_sec: float,
    ) -> Dict[str, float]:
        """统计一代种群的性能指标。"""

        if not population:
            return {
                "generation": float(generation),
                "elapsed_sec": float(elapsed_sec),
                "population_size": 0.0,
                "feasible_count": 0.0,
                "feasible_ratio": 0.0,
                "front0_size": 0.0,
                "best_weighted_total": float("inf"),
                "mean_weighted_total": float("inf"),
                "best_f1": float("inf"),
                "best_f2": float("inf"),
                "best_f3": float("inf"),
            }

        weighted_values: List[float] = []
        objective_values: List[Tuple[float, float, float]] = []
        feasible_flags: List[bool] = []

        for ind in population:
            if ind.decoded is None:
                self._evaluate_nsga2_individual(ind)
            decoded = ind.decoded if ind.decoded is not None else {}
            weighted_values.append(float(decoded.get("weighted_total", float("inf"))))
            objective_values.append(
                (
                    float(ind.objectives[0]),
                    float(ind.objectives[1]),
                    float(ind.objectives[2]),
                )
            )
            feasible_flags.append(self._is_feasible(ind))

        arr_w = np.asarray(weighted_values, dtype=float)
        arr_obj = np.asarray(objective_values, dtype=float)
        feasible_mask = np.asarray(feasible_flags, dtype=bool)

        use_idx = np.where(feasible_mask)[0]
        if use_idx.size == 0:
            use_idx = np.arange(len(population))

        use_w = arr_w[use_idx]
        use_obj = arr_obj[use_idx, :]

        fronts = self._fast_nondominated_sort(population)
        front0_size = len(fronts[0]) if fronts else 0

        return {
            "generation": float(generation),
            "elapsed_sec": float(elapsed_sec),
            "population_size": float(len(population)),
            "feasible_count": float(np.sum(feasible_mask)),
            "feasible_ratio": float(np.mean(feasible_mask)),
            "front0_size": float(front0_size),
            "best_weighted_total": float(np.min(use_w)),
            "mean_weighted_total": float(np.mean(use_w)),
            "best_f1": float(np.min(use_obj[:, 0])),
            "best_f2": float(np.min(use_obj[:, 1])),
            "best_f3": float(np.min(use_obj[:, 2])),
        }

    def _make_solution_from_decoded(
        self,
        decoded: Dict[str, Any],
        solver_mode: str,
        objective_vector: Optional[Tuple[float, float, float]],
        pareto_front: Optional[List[Dict[str, Any]]] = None,
    ) -> FusedModelSolution:
        order_plan = self._build_order_plan(
            assigned_candidate=decoded["assigned_candidate"],
            assigned_drone=decoded["assigned_drone"],
            assigned_rider=decoded["assigned_rider"],
            depart_time=decoded["depart_time"],
            arrival_time=decoded["arrival_time"],
            delivery_time=decoded["delivery_time"],
            lateness=decoded["lateness"],
        )

        return FusedModelSolution(
            selected_candidates=list(decoded["selected_candidates"]),
            order_plan=order_plan,
            drone_order_indices={int(k): list(v) for k, v in decoded["drone_order_indices"].items()},
            rider_order_indices={int(k): list(v) for k, v in decoded["rider_order_indices"].items()},
            objective_components=dict(decoded["components"]),
            total_objective=float(decoded["weighted_total"]),
            constraint_report=dict(decoded["constraint_report"]),
            solver_mode=solver_mode,
            objective_vector=objective_vector,
            pareto_front=pareto_front,
        )

    def _solve_heuristic(self) -> FusedModelSolution:
        """原有启发式单解流程（保持兼容）。"""

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

        decoded = {
            "selected_candidates": selected_candidates,
            "assigned_candidate": assigned_candidate,
            "assigned_drone": assigned_drone,
            "assigned_rider": assigned_rider,
            "depart_time": depart_time,
            "arrival_time": arrival_time,
            "delivery_time": delivery_time,
            "lateness": lateness,
            "drone_order_indices": drone_order_indices,
            "rider_order_indices": rider_order_indices,
            "drone_energy_used": drone_energy_used,
            "candidate_load": candidate_load,
            "constraint_report": constraint_report,
            "components": components,
            "weighted_total": total_objective,
        }

        objective_vector = (
            float(
                components["drone_cost"]
                + components["rider_spatiotemporal_cost"]
                + components["makespan"]
            ),
            float(components["late_cost"] + components["open_cost"]),
            float(components["drone_cost"]),
        )
        feasible_flag = bool(constraint_report.get("all_constraints_ok", False))
        self.last_performance_report = {
            "solver_mode": "heuristic",
            "population_size": 1.0,
            "generations": 1.0,
            "final_weighted_total": float(total_objective),
            "final_objective_vector": objective_vector,
            "generation_trace": [
                {
                    "generation": 1.0,
                    "elapsed_sec": 0.0,
                    "population_size": 1.0,
                    "feasible_count": 1.0 if feasible_flag else 0.0,
                    "feasible_ratio": 1.0 if feasible_flag else 0.0,
                    "front0_size": 1.0,
                    "best_weighted_total": float(total_objective),
                    "mean_weighted_total": float(total_objective),
                    "best_f1": float(objective_vector[0]),
                    "best_f2": float(objective_vector[1]),
                    "best_f3": float(objective_vector[2]),
                }
            ],
        }

        return self._make_solution_from_decoded(
            decoded=decoded,
            solver_mode="heuristic",
            objective_vector=objective_vector,
            pareto_front=None,
        )

    def _solve_nsga2(self) -> FusedModelSolution:
        """改进 NSGA-II 多目标优化流程。"""

        self._build_uav_matrices()
        self._build_order_candidate_score_matrix()

        pop_size = max(4, int(self.config.ga_population_size))
        n_gen = max(1, int(self.config.ga_generations))

        population = self._initialize_population()
        generation_trace: List[Dict[str, float]] = []
        start_ts = time.perf_counter()

        for gen in range(n_gen):
            fronts = self._fast_nondominated_sort(population)
            for front in fronts:
                self._assign_crowding_distance(population, front)

            offspring: List[NSGA2Individual] = []
            while len(offspring) < pop_size:
                p1 = self._tournament_pick(population)
                p2 = self._tournament_pick(population)

                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)

                self._evaluate_nsga2_individual(c1)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    self._evaluate_nsga2_individual(c2)
                    offspring.append(c2)

            merged = population + offspring
            population = self._environmental_selection(merged, pop_size)
            generation_trace.append(
                self._collect_generation_metrics(
                    population=population,
                    generation=gen + 1,
                    elapsed_sec=time.perf_counter() - start_ts,
                )
            )

            if self.config.ga_verbose and ((gen + 1) % 10 == 0 or gen == n_gen - 1):
                feasible_cnt = sum(1 for ind in population if self._is_feasible(ind))
                print(f"[NSGA-II] generation={gen + 1}/{n_gen}, feasible={feasible_cnt}/{len(population)}")

        final_fronts = self._fast_nondominated_sort(population)
        if not final_fronts or not final_fronts[0]:
            # 理论上不会发生，兜底返回启发式
            return self._solve_heuristic()

        front0 = final_fronts[0]
        feasible_front = [idx for idx in front0 if self._is_feasible(population[idx])]
        candidate_front = feasible_front if feasible_front else front0

        # 导出前沿时，基于最终候选前沿重算一次拥挤度，确保指标一致
        self._assign_crowding_distance(population, candidate_front)

        for idx in candidate_front:
            if population[idx].decoded is None:
                self._evaluate_nsga2_individual(population[idx])

        obj_arr = np.array([population[idx].objectives for idx in candidate_front], dtype=float)
        mins = np.min(obj_arr, axis=0)
        maxs = np.max(obj_arr, axis=0)
        span = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)

        w1 = max(1e-9, self.config.lambda_drone_cost + self.config.lambda_rider_cost + self.config.lambda_makespan)
        w2 = max(1e-9, self.config.lambda_late_cost + self.config.lambda_open_cost)
        w3 = max(1e-9, self.config.lambda_drone_cost)
        w = np.array([w1, w2, w3], dtype=float)
        w /= np.sum(w)

        best_idx = candidate_front[0]
        best_score = float("inf")
        for idx in candidate_front:
            norm_obj = (np.array(population[idx].objectives, dtype=float) - mins) / span
            score = float(np.dot(w, norm_obj))
            if score < best_score:
                best_score = score
                best_idx = idx

        best_ind = population[best_idx]
        if best_ind.decoded is None:
            self._evaluate_nsga2_individual(best_ind)
        assert best_ind.decoded is not None

        pareto_front: List[Dict[str, Any]] = []
        for idx in candidate_front:
            ind = population[idx]
            if ind.decoded is None:
                self._evaluate_nsga2_individual(ind)
            selected_count = 0
            weighted_total = float("inf")
            if ind.decoded is not None:
                selected_count = len(ind.decoded.get("selected_candidates", []))
                weighted_total = float(ind.decoded.get("weighted_total", float("inf")))
            pareto_front.append(
                {
                    "objective_vector": tuple(float(v) for v in ind.objectives),
                    "constraint_violation": float(ind.cv),
                    "rank": int(ind.rank),
                    "crowding_distance": float(ind.crowding_distance),
                    "weighted_total": weighted_total,
                    "selected_candidates_count": int(selected_count),
                }
            )

        self.last_performance_report = {
            "solver_mode": "nsga2",
            "population_size": float(pop_size),
            "generations": float(n_gen),
            "generation_trace": generation_trace,
            "final_front_size": float(len(candidate_front)),
            "selected_weighted_total": float(best_ind.decoded.get("weighted_total", float("inf"))),
            "selected_objective_vector": tuple(float(v) for v in best_ind.objectives),
        }

        return self._make_solution_from_decoded(
            decoded=best_ind.decoded,
            solver_mode="nsga2",
            objective_vector=tuple(float(v) for v in best_ind.objectives),
            pareto_front=pareto_front,
        )

    def _solve_classic_nsga(self) -> FusedModelSolution:
        """传统 NSGA 多目标优化流程（无拥挤距离，仅基于Pareto秩和共享）。"""

        self._build_uav_matrices()
        self._build_order_candidate_score_matrix()

        pop_size = max(4, int(self.config.ga_population_size))
        n_gen = max(1, int(self.config.ga_generations))
        sigma_share = pop_size // 4
        epsilon = 0.01

        population = self._initialize_population()
        generation_trace: List[Dict[str, float]] = []
        start_ts = time.perf_counter()

        def _sharing_distance(a: NSGA2Individual, b: NSGA2Individual) -> float:
            obj_arr = np.array([a.objectives, b.objectives], dtype=float)
            if obj_arr.shape[0] < 2:
                return 0.0
            d = np.linalg.norm(obj_arr[0] - obj_arr[1])
            return d

        def _niche_count(ind: NSGA2Individual, pop: List[NSGA2Individual]) -> float:
            count = 0.0
            for other in pop:
                if other is ind:
                    continue
                dist = _sharing_distance(ind, other)
                if dist < sigma_share:
                    count += 1.0 - (dist / sigma_share) ** epsilon
            return count

        for gen in range(n_gen):
            fronts = self._fast_nondominated_sort(population)
            for ind in population:
                if ind.decoded is None:
                    self._evaluate_nsga2_individual(ind)

            offspring: List[NSGA2Individual] = []
            while len(offspring) < pop_size:
                p1 = self._tournament_pick(population)
                p2 = self._tournament_pick(population)

                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)

                self._evaluate_nsga2_individual(c1)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    self._evaluate_nsga2_individual(c2)
                    offspring.append(c2)

            merged = population + offspring
            for ind in merged:
                if ind.decoded is None:
                    self._evaluate_nsga2_individual(ind)

            merged_fronts = self._fast_nondominated_sort(merged)
            new_pop: List[NSGA2Individual] = []

            for front in merged_fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend([merged[i] for i in front])
                else:
                    remaining = pop_size - len(new_pop)
                    for idx in front:
                        niche = _niche_count(merged[idx], [merged[i] for i in front])
                        merged[idx].crowding_distance = 1.0 / (niche + 1.0)

                    sorted_front = sorted(
                        front,
                        key=lambda i: merged[i].crowding_distance,
                        reverse=True,
                    )
                    new_pop.extend([merged[i] for i in sorted_front[:remaining]])
                    break

            population = new_pop
            generation_trace.append(
                self._collect_generation_metrics(
                    population=population,
                    generation=gen + 1,
                    elapsed_sec=time.perf_counter() - start_ts,
                )
            )

            if self.config.ga_verbose and ((gen + 1) % 10 == 0 or gen == n_gen - 1):
                feasible_cnt = sum(1 for ind in population if self._is_feasible(ind))
                print(f"[Classic NSGA] generation={gen + 1}/{n_gen}, feasible={feasible_cnt}/{len(population)}")

        final_fronts = self._fast_nondominated_sort(population)
        if not final_fronts or not final_fronts[0]:
            return self._solve_heuristic()

        front0 = final_fronts[0]
        feasible_front = [idx for idx in front0 if self._is_feasible(population[idx])]
        candidate_front = feasible_front if feasible_front else front0

        for idx in candidate_front:
            if population[idx].decoded is None:
                self._evaluate_nsga2_individual(population[idx])

        obj_arr = np.array([population[idx].objectives for idx in candidate_front], dtype=float)
        mins = np.min(obj_arr, axis=0)
        maxs = np.max(obj_arr, axis=0)
        span = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)

        w1 = max(1e-9, self.config.lambda_drone_cost + self.config.lambda_rider_cost + self.config.lambda_makespan)
        w2 = max(1e-9, self.config.lambda_late_cost + self.config.lambda_open_cost)
        w3 = max(1e-9, self.config.lambda_drone_cost)
        w = np.array([w1, w2, w3], dtype=float)
        w /= np.sum(w)

        best_idx = candidate_front[0]
        best_score = float("inf")
        for idx in candidate_front:
            norm_obj = (np.array(population[idx].objectives, dtype=float) - mins) / span
            score = float(np.dot(w, norm_obj))
            if score < best_score:
                best_score = score
                best_idx = idx

        best_ind = population[best_idx]
        if best_ind.decoded is None:
            self._evaluate_nsga2_individual(best_ind)
        assert best_ind.decoded is not None

        pareto_front: List[Dict[str, Any]] = []
        for idx in candidate_front:
            ind = population[idx]
            if ind.decoded is None:
                self._evaluate_nsga2_individual(ind)
            selected_count = 0
            weighted_total = float("inf")
            if ind.decoded is not None:
                selected_count = len(ind.decoded.get("selected_candidates", []))
                weighted_total = float(ind.decoded.get("weighted_total", float("inf")))
            pareto_front.append(
                {
                    "objective_vector": tuple(float(v) for v in ind.objectives),
                    "constraint_violation": float(ind.cv),
                    "rank": int(ind.rank),
                    "crowding_distance": float(ind.crowding_distance),
                    "weighted_total": weighted_total,
                    "selected_candidates_count": int(selected_count),
                }
            )

        self.last_performance_report = {
            "solver_mode": "classic_nsga",
            "population_size": float(pop_size),
            "generations": float(n_gen),
            "generation_trace": generation_trace,
            "final_front_size": float(len(candidate_front)),
            "selected_weighted_total": float(best_ind.decoded.get("weighted_total", float("inf"))),
            "selected_objective_vector": tuple(float(v) for v in best_ind.objectives),
        }

        return self._make_solution_from_decoded(
            decoded=best_ind.decoded,
            solver_mode="classic_nsga",
            objective_vector=tuple(float(v) for v in best_ind.objectives),
            pareto_front=pareto_front,
        )

    def _solve_classic_ga(self) -> FusedModelSolution:
        """经典遗传算法优化流程。"""
        
        self._build_uav_matrices()
        self._build_order_candidate_score_matrix()
        
        pop_size = max(4, int(self.config.ga_population_size))
        n_gen = max(1, int(self.config.ga_generations))
        
        # 初始化种群
        population = []
        for _ in range(pop_size):
            ind = self._random_individual()
            self._evaluate_nsga2_individual(ind)
            population.append(ind)
        
        generation_trace: List[Dict[str, float]] = []
        start_ts = time.perf_counter()
        
        for gen in range(n_gen):
            # 评估种群
            for ind in population:
                if ind.decoded is None:
                    self._evaluate_nsga2_individual(ind)
            
            # 选择（基于单目标适应度）
            fitness = [float(ind.decoded.get("weighted_total", float("inf"))) for ind in population]
            selected = []
            for _ in range(pop_size):
                # 轮盘赌选择
                weights = [1.0 / (f + 1e-9) for f in fitness]
                total_weight = sum(weights)
                r = self._rng.random() * total_weight
                current = 0
                for i, w in enumerate(weights):
                    current += w
                    if current >= r:
                        selected.append(population[i])
                        break
            
            # 交叉
            offspring = []
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size:
                    p1 = selected[i]
                    p2 = selected[i+1]
                    c1, c2 = self._crossover(p1, p2)
                    offspring.extend([c1, c2])
                else:
                    offspring.append(selected[i])
            
            # 变异
            for ind in offspring:
                self._mutate(ind)
                self._evaluate_nsga2_individual(ind)
            
            # 替换
            population = offspring
            
            # 收集指标
            generation_trace.append(
                self._collect_generation_metrics(
                    population=population,
                    generation=gen + 1,
                    elapsed_sec=time.perf_counter() - start_ts,
                )
            )
            
            if self.config.ga_verbose and ((gen + 1) % 10 == 0 or gen == n_gen - 1):
                feasible_cnt = sum(1 for ind in population if self._is_feasible(ind))
                print(f"[Classic GA] generation={gen + 1}/{n_gen}, feasible={feasible_cnt}/{len(population)}")
        
        # 选择最优个体
        best_ind = min(population, key=lambda ind: float(ind.decoded.get("weighted_total", float("inf")))
                      if ind.decoded is not None else float("inf"))
        
        if best_ind.decoded is None:
            # 兜底返回启发式
            return self._solve_heuristic()
        
        self.last_performance_report = {
            "solver_mode": "classic_ga",
            "population_size": float(pop_size),
            "generations": float(n_gen),
            "generation_trace": generation_trace,
            "final_weighted_total": float(best_ind.decoded.get("weighted_total", float("inf"))),
        }
        
        return self._make_solution_from_decoded(
            decoded=best_ind.decoded,
            solver_mode="classic_ga",
            objective_vector=None,
            pareto_front=None,
        )

    def _solve_pso(self) -> FusedModelSolution:
        """粒子群算法优化流程。"""
        
        self._build_uav_matrices()
        self._build_order_candidate_score_matrix()
        
        pop_size = max(4, int(self.config.ga_population_size))
        n_gen = max(1, int(self.config.ga_generations))
        
        # 粒子类
        class PSOParticle:
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.individual = optimizer._random_individual()
                self.optimizer._evaluate_nsga2_individual(self.individual)
                self.fitness = float(self.individual.decoded.get("weighted_total", float("inf")))
                self.pbest = self.individual
                self.pbest_fitness = self.fitness
                self.velocity = {
                    "order_sequence": [0.0] * len(self.individual.order_sequence),
                    "candidate_assignment": [0.0] * len(self.individual.candidate_assignment),
                    "candidate_open": [0.0] * len(self.individual.candidate_open)
                }
            
            def update_velocity(self, gbest, w=0.7, c1=1.4, c2=1.4):
                r1 = self.optimizer._rng.random()
                r2 = self.optimizer._rng.random()
                
                # 更新order_sequence速度
                for i in range(len(self.velocity["order_sequence"])):
                    self.velocity["order_sequence"][i] = (
                        w * self.velocity["order_sequence"][i] +
                        c1 * r1 * (self.pbest.order_sequence[i] - self.individual.order_sequence[i]) +
                        c2 * r2 * (gbest.order_sequence[i] - self.individual.order_sequence[i])
                    )
                
                # 更新candidate_assignment速度
                for i in range(len(self.velocity["candidate_assignment"])):
                    self.velocity["candidate_assignment"][i] = (
                        w * self.velocity["candidate_assignment"][i] +
                        c1 * r1 * (self.pbest.candidate_assignment[i] - self.individual.candidate_assignment[i]) +
                        c2 * r2 * (gbest.candidate_assignment[i] - self.individual.candidate_assignment[i])
                    )
                
                # 更新candidate_open速度
                for i in range(len(self.velocity["candidate_open"])):
                    self.velocity["candidate_open"][i] = (
                        w * self.velocity["candidate_open"][i] +
                        c1 * r1 * (self.pbest.candidate_open[i] - self.individual.candidate_open[i]) +
                        c2 * r2 * (gbest.candidate_open[i] - self.individual.candidate_open[i])
                    )
            
            def update_position(self):
                # 更新order_sequence
                new_order = self.individual.order_sequence.copy()
                for i in range(len(new_order)):
                    new_order[i] += self.velocity["order_sequence"][i]
                    new_order[i] = max(0, min(len(new_order)-1, int(new_order[i])))
                # 确保顺序唯一
                new_order = self._unique_order(new_order)
                self.individual.order_sequence = new_order
                
                # 更新candidate_assignment
                for i in range(len(self.individual.candidate_assignment)):
                    new_val = self.individual.candidate_assignment[i] + self.velocity["candidate_assignment"][i]
                    self.individual.candidate_assignment[i] = max(0, min(len(self.individual.candidate_open)-1, int(new_val)))
                
                # 更新candidate_open
                for i in range(len(self.individual.candidate_open)):
                    new_val = self.individual.candidate_open[i] + self.velocity["candidate_open"][i]
                    self.individual.candidate_open[i] = 1 if new_val > 0.5 else 0
                
                # 重新评估
                self.optimizer._evaluate_nsga2_individual(self.individual)
                self.fitness = float(self.individual.decoded.get("weighted_total", float("inf")))
                
                # 更新个人最优
                if self.fitness < self.pbest_fitness:
                    self.pbest = self.optimizer._clone_individual(self.individual)
                    self.pbest_fitness = self.fitness
            
            def _unique_order(self, order):
                # 确保顺序唯一
                unique = list(set(order))
                if len(unique) < len(order):
                    # 填充缺失的值
                    all_vals = set(range(len(order)))
                    missing = list(all_vals - set(unique))
                    self.optimizer._rng.shuffle(missing)
                    unique.extend(missing[:len(order)-len(unique)])
                # 打乱顺序以增加多样性
                self.optimizer._rng.shuffle(unique)
                return unique[:len(order)]
        
        # 初始化粒子群
        particles = [PSOParticle(self) for _ in range(pop_size)]
        
        # 找到全局最优
        gbest = min(particles, key=lambda p: p.fitness).pbest
        gbest_fitness = min(particles, key=lambda p: p.fitness).fitness
        
        generation_trace: List[Dict[str, float]] = []
        start_ts = time.perf_counter()
        
        for gen in range(n_gen):
            # 更新每个粒子
            for particle in particles:
                particle.update_velocity(gbest)
                particle.update_position()
            
            # 更新全局最优
            current_best = min(particles, key=lambda p: p.fitness)
            if current_best.fitness < gbest_fitness:
                gbest = current_best.pbest
                gbest_fitness = current_best.fitness
            
            # 收集指标
            population = [p.individual for p in particles]
            generation_trace.append(
                self._collect_generation_metrics(
                    population=population,
                    generation=gen + 1,
                    elapsed_sec=time.perf_counter() - start_ts,
                )
            )
            
            if self.config.ga_verbose and ((gen + 1) % 10 == 0 or gen == n_gen - 1):
                feasible_cnt = sum(1 for p in particles if self._is_feasible(p.individual))
                print(f"[PSO] generation={gen + 1}/{n_gen}, feasible={feasible_cnt}/{len(particles)}, best_fitness={gbest_fitness:.2f}")
        
        # 确保gbest已评估
        if gbest.decoded is None:
            self._evaluate_nsga2_individual(gbest)
        
        if gbest.decoded is None:
            # 兜底返回启发式
            return self._solve_heuristic()
        
        self.last_performance_report = {
            "solver_mode": "pso",
            "population_size": float(pop_size),
            "generations": float(n_gen),
            "generation_trace": generation_trace,
            "final_weighted_total": float(gbest.decoded.get("weighted_total", float("inf"))),
        }
        
        return self._make_solution_from_decoded(
            decoded=gbest.decoded,
            solver_mode="pso",
            objective_vector=None,
            pareto_front=None,
        )

    def solve(self) -> FusedModelSolution:
        """执行融合求解并返回结构化结果。"""

        self.last_performance_report = None
        mode = str(self.config.solver_mode).strip().lower()
        if mode in {"nsga2", "multi", "multi_objective", "multi-objective"}:
            return self._solve_nsga2()
        elif mode in {"classic_nsga", "nsga"}:
            return self._solve_classic_nsga()
        elif mode in {"classic_ga", "ga", "genetic"}:
            return self._solve_classic_ga()
        elif mode in {"pso", "particle_swarm"}:
            return self._solve_pso()
        return self._solve_heuristic()


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
    solver_mode: Optional[str] = None,
    render_plan: Union[bool, str, Sequence[str]] = False,
    render_save_path: Optional[str] = None,
    render_show: bool = False,
    render_performance: bool = False,
    performance_save_dir: Optional[str] = None,
    performance_show: bool = False,
    performance_file_prefix: str = "fusion_model",
    export_csv: bool = True,
    csv_output_dir: Optional[str] = None,
) -> FusedModelSolution:
    """融合模型的便捷入口函数。

    参数:
        render_plan:
            是否渲染路径图；支持:
            - bool: True=融合图，False=不渲染
            - str: "fused"/"uav"/"rider"/"both"/"all"
            - Sequence[str]: 例如 ["uav", "rider"]
        render_save_path:
            路径图输出路径；多图模式下:
            - 若传文件名（含后缀），自动在文件名后追加 "_uav/_rider/_fused"
            - 若传目录（不含后缀），写为 fusion_plan_*.png
            - 所有输出文件会自动追加 "_算法名" 后缀
        render_show: 是否弹出交互窗口显示路径图
        render_performance: 是否绘制性能对比图
        performance_save_dir: 性能图输出目录，默认 `./统一输出`
        performance_show: 是否弹出性能图窗口
        performance_file_prefix: 性能图文件名前缀（最终会自动追加 "_算法名"）
        export_csv: 是否导出三类 CSV 报表（文件名自动追加 "_算法名"）
        csv_output_dir: CSV 输出目录；为空时自动回退到 `performance_save_dir`、
            `render_save_path` 所在目录或 `./统一输出`
    """

    orders = build_orders_from_customers(
        customers=customers,
        merchants=merchants,
        order_count=order_count,
        random_seed=random_seed,
    )
    cfg = config if config is not None else FusionModelConfig()
    if solver_mode is not None:
        cfg.solver_mode = str(solver_mode)

    optimizer = DroneRiderFusionOptimizer(
        calculator=calculator,
        merchants=merchants,
        candidate_points=candidate_points,
        orders=orders,
        n_drones=n_drones,
        n_riders=n_riders,
        config=cfg,
    )
    t_opt = time.perf_counter()
    solution = optimizer.solve()
    algorithm_name = str(solution.solver_mode or cfg.solver_mode or "unknown")
    optimize_elapsed = time.perf_counter() - t_opt
    print(f"[Solve] Optimization finished, elapsed={optimize_elapsed:.1f}s")
    if optimizer.last_performance_report is not None:
        solution.performance_report = dict(optimizer.last_performance_report)

    render_modes = _normalize_render_plan_modes(render_plan)
    if not render_modes and render_save_path:
        render_modes = ["fused"]

    if render_modes:
        print(f"[Post] Rendering route maps: modes={render_modes}")
        save_path_map = _resolve_render_output_paths(
            render_save_path,
            render_modes,
            algorithm_name=algorithm_name,
        )
        rendered_paths: Dict[str, str] = {}
        for mode in render_modes:
            t_render = time.perf_counter()
            print(f"[Post] Start render mode={mode} ...")
            show_uav, show_rider, title = _render_mode_to_flags(mode)
            mode_save_path = save_path_map.get(mode)
            saved = render_fused_solution_map(
                terrain=calculator.terrain,
                obstacles=calculator.obstacles,
                merchants=merchants,
                customers=customers,
                candidate_points=candidate_points,
                solution=solution,
                save_path=mode_save_path,
                show=(render_show or mode_save_path is None),
                safe_clearance_height=optimizer.config.safe_clearance_height,
                min_flight_height=optimizer.config.min_flight_height,
                max_flight_height=optimizer.config.max_flight_height,
                enable_multi_waypoint_search=optimizer.config.enable_multi_waypoint_search,
                path_grid_step_m=optimizer.config.path_grid_step_m,
                path_search_margin_m=optimizer.config.path_search_margin_m,
                path_obstacle_buffer_m=optimizer.config.path_obstacle_buffer_m,
                path_max_expand_nodes=optimizer.config.path_max_expand_nodes,
                path_max_waypoints=optimizer.config.path_max_waypoints,
                show_uav_paths=show_uav,
                show_rider_paths=show_rider,
                title=title,
            )
            print(f"[Post] Done render mode={mode}, elapsed={time.perf_counter() - t_render:.1f}s")
            if saved:
                rendered_paths[mode] = saved
        if rendered_paths:
            solution.rendered_plan_paths = rendered_paths

    if render_performance or performance_save_dir:
        t_perf = time.perf_counter()
        print("[Post] Rendering performance plots ...")
        performance_paths = render_fused_performance_plots(
            solution=solution,
            save_dir=performance_save_dir,
            show=performance_show,
            file_prefix=performance_file_prefix,
            algorithm_name=algorithm_name,
        )
        print(f"[Post] Done performance plots, elapsed={time.perf_counter() - t_perf:.1f}s")
        if performance_paths:
            solution.performance_plot_paths = performance_paths

    if export_csv:
        t_csv = time.perf_counter()
        print("[Post] Exporting CSV reports ...")
        tabular_dir = _resolve_tabular_output_dir(
            csv_output_dir=csv_output_dir,
            render_save_path=render_save_path,
            performance_save_dir=performance_save_dir,
        )
        csv_paths = export_fused_solution_csv_reports(
            calculator=calculator,
            merchants=merchants,
            customers=customers,
            candidate_points=candidate_points,
            orders=orders,
            solution=solution,
            output_dir=tabular_dir,
            rider_count_for_rider_only=optimizer.n_riders,
            rider_capacity_for_rider_only=optimizer.config.rider_capacity,
            rider_speed_for_rider_only_kmh=optimizer.config.rider_speed_rider_only_kmh,
            safe_clearance_height=optimizer.config.safe_clearance_height,
            min_flight_height=optimizer.config.min_flight_height,
            max_flight_height=optimizer.config.max_flight_height,
            enable_multi_waypoint_search=optimizer.config.enable_multi_waypoint_search,
            path_grid_step_m=optimizer.config.path_grid_step_m,
            path_search_margin_m=optimizer.config.path_search_margin_m,
            path_obstacle_buffer_m=optimizer.config.path_obstacle_buffer_m,
            path_max_expand_nodes=optimizer.config.path_max_expand_nodes,
            path_max_waypoints=optimizer.config.path_max_waypoints,
            algorithm_name=algorithm_name,
        )
        print(f"[Post] Done CSV reports, elapsed={time.perf_counter() - t_csv:.1f}s")
        if csv_paths:
            solution.csv_output_paths = csv_paths

    return solution


if __name__ == "__main__":
    main()
