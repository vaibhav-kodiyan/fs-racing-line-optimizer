"""Tests for utility functions."""

import pytest
import numpy as np
import json
import tempfile
import os
from fmsim.utils import load_cones_json, car_triangle


class TestLoadConesJson:
    """Test cone loading functionality."""

    def test_load_valid_cones(self):
        """Test loading valid cone data."""
        test_data = {
            "left": [[0, 1], [1, 1.1], [2, 0.9]],
            "right": [[0, -1], [1, -0.9], [2, -1.1]]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            left, right = load_cones_json(temp_path)

            assert len(left) == 3
            assert len(right) == 3
            assert left.dtype == float
            assert right.dtype == float

            np.testing.assert_array_equal(left, np.array([[0, 1], [1, 1.1], [2, 0.9]]))
            np.testing.assert_array_equal(right, np.array([[0, -1], [1, -0.9], [2, -1.1]]))
        finally:
            os.unlink(temp_path)

    def test_load_empty_cones(self):
        """Test loading with missing cone data."""
        test_data = {"left": [], "right": []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            left, right = load_cones_json(temp_path)

            assert len(left) == 0
            assert len(right) == 0
            assert left.dtype == float
            assert right.dtype == float
        finally:
            os.unlink(temp_path)

    def test_load_partial_cones(self):
        """Test loading with only one side of cones."""
        test_data = {"left": [[0, 1], [1, 1]], "right": []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            left, right = load_cones_json(temp_path)

            assert len(left) == 2
            assert len(right) == 0
            np.testing.assert_array_equal(left, np.array([[0, 1], [1, 1]]))
        finally:
            os.unlink(temp_path)

    def test_load_missing_keys(self):
        """Test loading with missing keys."""
        test_data = {"other_data": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            left, right = load_cones_json(temp_path)

            assert len(left) == 0
            assert len(right) == 0
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_cones_json("nonexistent_file.json")


class TestCarTriangle:
    """Test car triangle geometry function."""

    def test_car_triangle_basic(self):
        """Test basic car triangle generation."""
        x, y, yaw = 0.0, 0.0, 0.0
        triangle = car_triangle(x, y, yaw)

        assert triangle.shape == (4, 2)
        assert isinstance(triangle, np.ndarray)

        # First point should be the front of the car
        assert triangle[0, 0] > triangle[1, 0]  # Front is ahead
        assert triangle[0, 0] > triangle[2, 0]

    def test_car_triangle_translation(self):
        """Test car triangle with translation."""
        x, y, yaw = 5.0, 3.0, 0.0
        triangle = car_triangle(x, y, yaw)

        # All points should be translated
        assert np.all(triangle[:, 0] >= 4.0)  # x offset
        assert np.all(triangle[:, 1] >= 2.0)  # y offset

    def test_car_triangle_rotation(self):
        """Test car triangle with rotation."""
        x, y, yaw = 0.0, 0.0, np.pi/2  # 90 degrees
        triangle = car_triangle(x, y, yaw)

        # After 90-degree rotation, front should point in +y direction
        assert triangle[0, 1] > 0  # Front y should be positive
        assert abs(triangle[0, 0]) < 0.1  # Front x should be near zero

    def test_car_triangle_scale(self):
        """Test car triangle with different scales."""
        x, y, yaw = 0.0, 0.0, 0.0

        triangle_small = car_triangle(x, y, yaw, scale=0.5)
        triangle_large = car_triangle(x, y, yaw, scale=2.0)

        # Larger scale should produce larger triangle
        small_size = np.max(np.linalg.norm(triangle_small, axis=1))
        large_size = np.max(np.linalg.norm(triangle_large, axis=1))

        assert large_size > small_size

    def test_car_triangle_closed_loop(self):
        """Test that triangle forms a closed loop."""
        x, y, yaw = 1.0, 2.0, np.pi/4
        triangle = car_triangle(x, y, yaw)

        # First and last points should be the same (closed triangle)
        np.testing.assert_array_almost_equal(triangle[0], triangle[-1])

    def test_car_triangle_symmetry(self):
        """Test triangle symmetry about centerline."""
        x, y, yaw = 0.0, 0.0, 0.0
        triangle = car_triangle(x, y, yaw)

        # Points 1 and 2 should be symmetric about x-axis
        assert abs(triangle[1, 1] + triangle[2, 1]) < 1e-10
        assert abs(triangle[1, 0] - triangle[2, 0]) < 1e-10

    def test_car_triangle_multiple_orientations(self):
        """Test car triangle at various orientations."""
        orientations = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]

        for yaw in orientations:
            triangle = car_triangle(0, 0, yaw)

            # Triangle should always have 4 points
            assert triangle.shape == (4, 2)

            # Triangle should be closed
            np.testing.assert_array_almost_equal(triangle[0], triangle[-1])

            # Check that rotation is applied correctly
            # Front point direction should match yaw
            front_angle = np.arctan2(triangle[0, 1], triangle[0, 0])
            assert abs(front_angle - yaw) < 1e-10 or abs(front_angle - yaw + 2*np.pi) < 1e-10
