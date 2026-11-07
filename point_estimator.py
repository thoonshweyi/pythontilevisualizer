"""Estimate four wall perspective points from a segmentation mask and image lines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class _QuadCandidate:
    points: np.ndarray
    method: str


def _order_points(points: np.ndarray) -> np.ndarray:
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _mask_quad(mask: np.ndarray) -> tuple[_QuadCandidate, np.ndarray, float]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Mask does not contain a foreground region")

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < 16:
        raise ValueError("Mask foreground region is too small")

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = min(1.0, contour_area / hull_area)
    perimeter = cv2.arcLength(hull, True)

    # Increasing epsilon removes small notches while retaining perspective.
    for epsilon_ratio in np.linspace(0.005, 0.07, 14):
        polygon = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(polygon) == 4:
            return (
                _QuadCandidate(_order_points(polygon[:, 0, :]), "mask_convex_hull"),
                contour,
                solidity,
            )

    rectangle = cv2.boxPoints(cv2.minAreaRect(hull))
    return _QuadCandidate(_order_points(rectangle), "min_area_rect"), contour, solidity


def _line_equation(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return None
    a = float(p1[1] - p2[1])
    b = float(p2[0] - p1[0])
    c = float(p1[0] * p2[1] - p2[0] * p1[1])
    norm = math.hypot(a, b)
    return np.array([a / norm, b / norm, c / norm], dtype=np.float64)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    determinant = first[0] * second[1] - second[0] * first[1]
    if abs(determinant) < 1e-6:
        return None
    x = (first[1] * second[2] - second[1] * first[2]) / determinant
    y = (first[2] * second[0] - second[2] * first[0]) / determinant
    return np.array([x, y], dtype=np.float32)


def _detect_side_lines(
    image_bgr: np.ndarray,
    contour: np.ndarray,
    quad: np.ndarray,
) -> list[np.ndarray | None]:
    height, width = image_bgr.shape[:2]
    diagonal = math.hypot(width, height)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = max(25, int(0.66 * median))
    high = max(low + 30, min(255, int(1.33 * median)))
    edges = cv2.Canny(gray, low, high)

    boundary = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(boundary, [contour], -1, 255, 2)
    tolerance = max(7, int(diagonal * 0.018))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    near_boundary = cv2.dilate(boundary, kernel)
    relevant_edges = cv2.bitwise_and(edges, near_boundary)

    lines = cv2.HoughLinesP(
        relevant_edges,
        1,
        np.pi / 180,
        threshold=max(20, int(diagonal * 0.035)),
        minLineLength=max(25, int(diagonal * 0.07)),
        maxLineGap=max(8, int(diagonal * 0.025)),
    )
    if lines is None:
        return [None, None, None, None]

    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    selected: list[np.ndarray | None] = [None, None, None, None]
    best_scores = np.zeros(4, dtype=np.float64)
    max_distance = max(12.0, diagonal * 0.055)

    # OpenCV versions return either (N, 1, 4) or (N, 4).
    for raw_line in np.asarray(lines).reshape(-1, 4):
        p1 = raw_line[:2].astype(np.float32)
        p2 = raw_line[2:].astype(np.float32)
        segment = p2 - p1
        length = float(np.linalg.norm(segment))
        if length < 1.0:
            continue
        segment_direction = segment / length
        equation = _line_equation(p1, p2)
        if equation is None:
            continue

        midpoint = (p1 + p2) / 2.0
        for side, (start_index, end_index) in enumerate(edge_pairs):
            start = quad[start_index]
            end = quad[end_index]
            candidate_direction = end - start
            candidate_length = float(np.linalg.norm(candidate_direction))
            if candidate_length < 1.0:
                continue
            candidate_direction /= candidate_length
            alignment = abs(float(np.dot(segment_direction, candidate_direction)))
            if alignment < 0.82:
                continue
            candidate_line = _line_equation(start, end)
            if candidate_line is None:
                continue
            distance = abs(float(candidate_line[0] * midpoint[0] + candidate_line[1] * midpoint[1] + candidate_line[2]))
            if distance > max_distance:
                continue
            score = length * alignment * (1.0 - distance / max_distance)
            if score > best_scores[side]:
                best_scores[side] = score
                selected[side] = equation

    return selected


def estimate_wall_points(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    segmentation_score: float | None = None,
) -> dict:
    """Estimate TL, TR, BR, BL points and confidence for one wall mask."""
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("A BGR color image is required")
    if mask is None or mask.ndim != 2:
        raise ValueError("A single-channel mask is required")
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions must match")

    candidate, contour, solidity = _mask_quad(mask)
    original = candidate.points
    side_lines = _detect_side_lines(image_bgr, contour, original)
    final = original.copy()
    height, width = mask.shape
    diagonal = math.hypot(width, height)
    snap_distance = max(18.0, diagonal * 0.12)
    accepted_intersections = 0

    # Each corner is the intersection of its preceding and following side.
    corner_sides = [(3, 0), (0, 1), (1, 2), (2, 3)]
    for corner_index, (first_side, second_side) in enumerate(corner_sides):
        first = side_lines[first_side]
        second = side_lines[second_side]
        if first is None or second is None:
            continue
        point = _intersection(first, second)
        if point is None:
            continue
        inside_margin = (
            -0.05 * width <= point[0] <= 1.05 * width
            and -0.05 * height <= point[1] <= 1.05 * height
        )
        if inside_margin and np.linalg.norm(point - original[corner_index]) <= snap_distance:
            final[corner_index] = point
            accepted_intersections += 1

    detected_sides = sum(line is not None for line in side_lines)
    used_lines = detected_sides >= 2 and accepted_intersections > 0
    method = "mask_and_lines" if used_lines else candidate.method

    geometry_confidence = min(
        0.95,
        0.42 + 0.18 * solidity + 0.055 * detected_sides + 0.04 * accepted_intersections,
    )
    if segmentation_score is None:
        confidence = geometry_confidence
    else:
        confidence = 0.55 * float(np.clip(segmentation_score, 0.0, 1.0)) + 0.45 * geometry_confidence

    points = [
        {
            "x": int(np.clip(round(point[0]), 0, width - 1)),
            "y": int(np.clip(round(point[1]), 0, height - 1)),
        }
        for point in final
    ]
    return {
        "estimated_points": points,
        "confidence": round(float(confidence), 4),
        "method": method,
        "detected_line_count": detected_sides,
    }
