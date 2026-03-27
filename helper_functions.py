import numpy as np
import math
from shapely.geometry import box, Point, LineString, Polygon
from shapely import affinity
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.path import make_path
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize, linemerge
import os

def scoring_function(angles, magnitudes, multi_output=False, joint_range=90):
    """
    angles:      shape (N,num sensors) -> angles[:,0] sensor1, angles[:,1] sensor2 ... in degrees
    magnitudes:  shape (N,num sensors) -> same layout in mT
    joint_range: specify how long the joint range is
    """
    magnitudes = magnitudes * 1000 # convert to micro T from mT
    num_angle_steps = len(angles[:,0])
    angle_scale = num_angle_steps/joint_range

    diffs= np.diff(angles, axis=0)

    diffs = (diffs + 180) % 360 - 180

    mag_avgs = (magnitudes[:-1,:] + magnitudes[1:, :]) /2
    weighted_angle_diffs = np.abs(np.deg2rad(diffs) * mag_avgs) # convert to rads to get arc lengths
    weighted_angle_diffs *= angle_scale # keeps it nice independent of num angle steps
    sum_weighted_angle_diffs = np.sum(weighted_angle_diffs,axis=1) # sum over the sensors
    score_min = min(sum_weighted_angle_diffs) # reaches around 7 on MCP_new, average reaches around 12
    score_avg = np.average(sum_weighted_angle_diffs)
    score = score_min + 0.2 * score_avg
    if multi_output:
        return score_min, score_avg, score
    else:
        return score_min


def calculate_overlap_area(pos1, pos2, size):
    """
    Calculates the exact area of overlap between two rotated squares.

    Args:
        pos1: Tuple (x, y, rotation_degrees) for the first square.
        pos2: Tuple (x, y, rotation_degrees) for the second square.
        size: Side length of the squares (float).

    Returns:
        float: The area of overlap (0.0 if no overlap).
    """
    x1, y1, rot1 = pos1
    x2, y2, rot2 = pos2

    # 1. Create a base square centered at (0,0)
    # box(minx, miny, maxx, maxy)
    base_sq = box(-size / 2, -size / 2, size / 2, size / 2)

    # 2. Transform Square 1
    # Rotate first (around center), then Translate
    poly1 = affinity.rotate(base_sq, rot1, origin='center')
    poly1 = affinity.translate(poly1, x1, y1)

    # 3. Transform Square 2
    poly2 = affinity.rotate(base_sq, rot2, origin='center')
    poly2 = affinity.translate(poly2, x2, y2)

    # 4. Calculate Intersection
    # If they don't overlap, intersection returns an empty geometry with area 0.0
    intersection = poly1.intersection(poly2)
    area = intersection.area

    # 5. Tolerance Filter (Optional)
    # This prevents floating point noise from flagging "touching" as "colliding"
    if area < 1e-12:
        return 0.0
    return area

def calculate_area_outside_circle(square_pos, square_size, circle_radius, circle_center=(0, 0)):
    """
    Calculates the area of a square that lies OUTSIDE a given circle.
    Useful for constraints (e.g., ensuring a magnet stays strictly inside a housing).

    Args:
        square_pos (tuple): (x, y, rotation_degrees) of the square's center.
        square_size (float): Side length of the square.
        circle_radius (float): Radius of the boundary circle.
        circle_center (tuple): (x, y) center of the circle. Default is (0,0).

    Returns:
        float: The area of the square sticking out of the circle.
               Returns 0.0 if the square is completely inside.
    """
    x, y, rot = square_pos

    # 1. Create the Circle
    # Shapely approximates circles as polygons. 'resolution' determines smoothness.
    # resolution=16 means 16 points per quarter-circle (64 total), usually sufficient.
    circle = Point(circle_center).buffer(circle_radius, resolution=32)

    # 2. Create the Square (centered at 0,0 first)
    # box(minx, miny, maxx, maxy)
    base_sq = box(-square_size / 2, -square_size / 2, square_size / 2, square_size / 2)

    # 3. Rotate and Translate Square to actual position
    # Rotate first around center
    square_poly = affinity.rotate(base_sq, rot, origin='center')
    # Translate to final (x,y)
    square_poly = affinity.translate(square_poly, x, y)

    # 4. Calculate Difference (Square - Circle)
    # This leaves only the parts of the square that are NOT inside the circle.
    diff_poly = square_poly.difference(circle)

    return diff_poly.area



def load_shape_from_dxf(filepath, curve_resolution=0.01, precision=4):
    """
    Loads a 2D shape from a DXF file, handling SolidWorks floating-point gaps.

    Parameters:
    - filepath: Path to the DXF file.
    - curve_resolution: The maximum allowed distance between curve and line.
    - precision: Number of decimal places to round coordinates to (fixes microscopic gaps).
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    line_segments = []

    # 1. Extract all geometry and convert to simple line strings
    for entity in msp:
        if entity.dxftype() in ['LINE', 'ARC', 'CIRCLE', 'SPLINE', 'LWPOLYLINE', 'POLYLINE']:
            try:
                path = make_path(entity)
                vertices = list(path.flattening(distance=curve_resolution))

                if len(vertices) >= 2:
                    # ROUNDING: This is the crucial fix for SolidWorks exports
                    coords = [(round(v.x, precision), round(v.y, precision)) for v in vertices]

                    # Ignore zero-length lines created by heavy rounding
                    if coords[0] != coords[-1] or len(coords) > 2:
                        line_segments.append(LineString(coords))
            except Exception:
                pass

    # 2. Merge overlapping or touching lines before polygonizing
    # linemerge turns connected segments into longer Continuous LineStrings
    merged_lines = linemerge(line_segments)

    # 3. Stitch the lines together to form closed polygons
    # polygonize expects a sequence of geometries, so we pass it inside a list if it's a single geometry
    if merged_lines.geom_type == 'LineString':
        geoms_to_polygonize = [merged_lines]
    else:
        geoms_to_polygonize = merged_lines.geoms

    polygons = list(polygonize(geoms_to_polygonize))

    if not polygons:
        raise ValueError(
            "Could not form a closed loop from the DXF data. "
            "Try decreasing the 'precision' parameter to snap larger gaps together."
        )

    # 4. Handle multiple loops
    main_shape = max(polygons, key=lambda p: p.area)

    return main_shape


def calculate_outside_area(base_shape, square_size, center_x, center_y, angle_deg, clearance=0.0):
    """
    Calculates how much of a square is outside a given base shape,
    allowing for an inward clearance offset.

    Parameters:
    - base_shape: The Shapely Polygon from SolidWorks.
    - square_size: The side length of the square.
    - center_x, center_y: The X and Y coordinates of the square's center.
    - angle_deg: Rotation angle of the square in degrees.
    - clearance: The distance to shrink the base_shape inwards.

    Returns:
    - outside_area: The total area of the square that falls outside.
    - percentage_outside: The percentage of the square outside (0-100).
    - placed_square: The Shapely Polygon of the final positioned square.
    - working_shape: The Shapely Polygon of the base shape (with clearance applied).
    """

    # 1. Apply the clearance (shift the boundary inwards)
    if clearance > 0:
        # A negative buffer shrinks the polygon perimeter inwards
        working_shape = base_shape.buffer(-clearance)

        # Safety check: if you shrink it too much, it disappears!
        if working_shape.is_empty:
            raise ValueError("The clearance is too large; the base shape collapsed to nothing.")
    else:
        working_shape = base_shape

    # 2. Create a base square centered at (0, 0)
    half_size = square_size / 2.0
    square = box(-half_size, -half_size, half_size, half_size)

    # 3. Rotate the square around its center
    rotated_square = affinity.rotate(square, angle_deg, origin='centroid')

    # 4. Translate the square to the target position
    placed_square = affinity.translate(rotated_square, xoff=center_x, yoff=center_y)

    # 5. Calculate the difference (Square MINUS Working Shape)
    outside_geometry = placed_square.difference(working_shape)

    # 6. Calculate areas
    outside_area = outside_geometry.area

    # Avoid division by zero if square_size happens to be 0
    square_area = placed_square.area
    percentage_outside = (outside_area / square_area) * 100 if square_area > 0 else 0

    return outside_area

def scale_and_rotate_dxf(base_shape, angle_deg=90, scale_factor=0.001):
    """
    Rotates the shape and shrinks it, strictly anchoring to the (0, 0) origin.
    """

    # 1. Rotate by the specified angle (positive is counter-clockwise)
    # By setting origin=(0,0), the shape orbits the origin rather than spinning in place.
    rotated_shape = affinity.rotate(base_shape, angle_deg, origin=(0, 0))

    # 2. Scale the shape
    # Shrinking by a factor of 1000 means multiplying by 0.001.
    # Anchoring to (0,0) ensures the shape pulls precisely toward the origin as it shrinks.
    final_shape = affinity.scale(
        rotated_shape,
        xfact=scale_factor,
        yfact=scale_factor,
        origin=(0, 0)
    )

    return final_shape


def mirror_and_shift_dxf(poly, origin_shift=0.006):
    # 1. Mirror across the X-axis (scale Y by -1)
    mirrored_poly = affinity.scale(poly, xfact=1.0, yfact=-1.0, origin=(0, 0))

    # 2. Shift coordinates as if the origin moved right
    # (Translate the shape itself to the left)
    final_poly = affinity.translate(mirrored_poly, xoff=-origin_shift, yoff=0.0)

    return final_poly


def save_results_to_csv(res, filename):
    """
    Saves the optimization results (Scores and Variables) to a CSV file.
    """
    if res.X is None or res.F is None:
        print(f"Skipping {filename}: No valid solutions to save.")
        return

    # 1. Force 2D arrays (in case only 1 solution was found)
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)

    # 2. Invert Scores back to positive (assuming you returned -score in _evaluate)
    positive_F = -F

    # 3. Combine Scores and Variables into one large block of data
    # Shape becomes: [Obj1, Obj2, Mag1_x, Mag1_y, Mag1_rot, Mag2_x, Mag2_y, Mag2_rot]
    combined_data = np.hstack((positive_F, X))

    # 4. Create a readable header
    header = "Score_Obj1, Score_Obj2, Mag1_X(m), Mag1_Y(m), Mag1_Rot(deg), Mag2_X(m), Mag2_Y(m), Mag2_Rot(deg)"

    # 5. Save to file
    np.savetxt(
        filename,
        combined_data,
        delimiter=",",
        header=header,
        comments="",  # Removes the '#' at the start of the header line
        fmt="%.6f"  # Keeps 6 decimal places for precision
    )

    print(f"Saved {len(X)} solutions to {filename}")





def visual_debugger(base_shape):
    """
    Plots the base shape, the test square, and marks the (0,0) origin.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Plot the Base Shape (DXF)
    # We use exterior.xy to get the outline coordinates
    x_base, y_base = base_shape.exterior.xy
    ax.plot(x_base, y_base, color='blue', linewidth=2, label='Base Shape (DXF)')
    ax.fill(x_base, y_base, alpha=0.2, color='blue')  # Light blue fill

    # 2. Plot the Test Square
    x_sq, y_sq = 0.003, 0.003
    ax.plot(x_sq, y_sq, color='red', linewidth=2, linestyle='--', label='Test Square')
    ax.fill(x_sq, y_sq, alpha=0.2, color='red')  # Light red fill

    # 3. Clearly mark the (0,0) Origin
    ax.plot(0, 0, marker='+', color='black', markersize=20, markeredgewidth=3, label='Origin (0,0)')

    # Formatting to make it look like a CAD grid
    ax.set_aspect('equal')  # Prevents distortion (squares will look like squares)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)  # X axis
    ax.axvline(0, color='black', linewidth=0.5)  # Y axis

    ax.legend()
    plt.title("Geometry Overlap Debugger")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


















# only for MCP board sensors
def convert_to_pcb_coord(base_x, base_y, raw_x, raw_y):
    pcb_x = base_x + raw_y
    pcb_y = base_y - raw_x
    return pcb_x, pcb_y

def check_collision(pos1, pos2, width, height):
    """
    Checks if two rotated rectangles overlap using the Separating Axis Theorem (SAT).

    Args:
        pos1: Tuple (x, y, rotation_in_degrees) for the first magnet.
        pos2: Tuple (x, y, rotation_in_degrees) for the second magnet.
        width: Width of the magnets.
        height: Height of the magnets.

    Returns:
        True if they overlap, False if they are separate.
    """

    # 1. Helper function to get the 4 corners of a rotated rectangle
    def get_corners(x, y, rot_deg, w, h):
        # Convert degrees to radians
        rad = np.radians(rot_deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # Half dimensions relative to center
        hw = w / 2.0
        hh = h / 2.0

        # Relative corners (unrotated)
        # Top-Right, Top-Left, Bottom-Left, Bottom-Right
        rel_corners = [
            (hw, hh),
            (-hw, hh),
            (-hw, -hh),
            (hw, -hh)
        ]

        # Rotate and translate corners to world space
        corners = []
        for rx, ry in rel_corners:
            # Rotation formula
            rotated_x = rx * cos_a - ry * sin_a
            rotated_y = rx * sin_a + ry * cos_a
            # Translate
            corners.append([x + rotated_x, y + rotated_y])

        return np.array(corners)

    # 2. Helper to project shape onto an axis
    def project_shape(corners, axis):
        # Dot product of all corners with the axis
        dots = np.dot(corners, axis)
        return np.min(dots), np.max(dots)

    # --- Main SAT Logic ---

    c1 = get_corners(pos1[0], pos1[1], pos1[2], width, height)
    c2 = get_corners(pos2[0], pos2[1], pos2[2], width, height)

    # We need to test 4 axes:
    # Two perpendicular to the edges of Rect 1, Two perpendicular to the edges of Rect 2.
    # These axes are just the edge vectors of the rectangles.

    # Edges of Rect 1 (only need two adjacent edges to get normals)
    edges1 = [c1[1] - c1[0], c1[1] - c1[2]]
    # Edges of Rect 2
    edges2 = [c2[1] - c2[0], c2[1] - c2[2]]

    # Combine edges to test (we use edges as the normal axes)
    axes_to_test = edges1 + edges2

    for axis in axes_to_test:
        # Normalize the axis is not strictly necessary for overlap check,
        # but good practice if you need projection depth.
        # Here we just check for gaps, so raw edge vectors work fine.

        # Project both rectangles onto this axis
        min1, max1 = project_shape(c1, axis)
        min2, max2 = project_shape(c2, axis)

        # Check for a gap
        if max1 < min2 or max2 < min1:
            return False  # Found a separating axis -> No collision!

    return True  # No separating axis found -> Collision detected

def check_possible_pos(xm, ym, rot, width, r_outer):
    """
    Checks if a square magnet of side magnet_size,
    centered at (xm, ym) and rotated by rot_deg degrees,
    lies fully inside a circle of radius r_outer.

    Circle center is (0,0).

    Width including clearance
    """

    # Convert rotation to radians
    rot = math.radians(rot)

    # Half the side length
    s = (width) / 2.0

    # Square corners in local coordinates (unrotated)
    corners_local = [
        (s, s),
        (s, -s),
        (-s, -s),
        (-s, s)
    ]

    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    for cx, cy in corners_local:
        # Rotate
        x_rot = cx * cos_r - cy * sin_r
        y_rot = cx * sin_r + cy * cos_r

        # Translate to global coordinates
        xg = xm + x_rot
        yg = ym + y_rot

        # Check inside circle
        if xg * xg + yg * yg > r_outer * r_outer:
            return False

    return True
