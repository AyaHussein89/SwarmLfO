import random
import math
from typing import List, Tuple

Point = Tuple[float, float]

def dist(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def circle_center(bx, by, cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return (cy * B - by * C) / (2 * D), (bx * C - cx * B) / (2 * D)

def circle_from(p1: Point, p2: Point, p3: Point):
    cx, cy = circle_center(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1])
    cx += p1[0]
    cy += p1[1]
    radius = dist((cx, cy), p1)
    return (cx, cy), radius

def circle_from_two(p1: Point, p2: Point):
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    radius = dist(p1, p2) / 2
    return center, radius

def is_in_circle(p: Point, c: Point, r: float) -> bool:
    return dist(p, c) <= r + 1e-9

def welzl(P: List[Point], R: List[Point]) -> Tuple[Point, float]:
    if not P or len(R) == 3:
        if len(R) == 0:
            return (0, 0), 0
        elif len(R) == 1:
            return R[0], 0
        elif len(R) == 2:
            return circle_from_two(R[0], R[1])
        else:
            return circle_from(R[0], R[1], R[2])
    p = P.pop()
    c, r = welzl(P, R)
    if is_in_circle(p, c, r):
        P.append(p)
        return c, r
    R.append(p)
    c, r = welzl(P, R)
    P.append(p)
    R.pop()
    return c, r

def smallest_enclosing_circle(points: List[Point]) -> Tuple[Point, float]:
    P = points[:]
    random.shuffle(P)
    return welzl(P, [])


