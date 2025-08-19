"""Extended tests for planner functionality."""

import pytest
import numpy as np
from fmsim.planner import (
    pair_cones_to_midline, _hungarian_pair_cones, _greedy_pair_cones,
    spline_smooth, optimization_based_racing_line, curvature_geometric,
    _apply_corridor_constraints, _interpolate_boundary
)


class TestHungarianPairing:
    """Test Hungarian algorithm cone pairing."""
    
    def test_hungarian_equal_cones(self):
        """Test Hungarian pairing with equal number of cones."""
        left = np.array([[0, 1], [1, 1], [2, 1]])
        right = np.array([[0, -1], [1, -1], [2, -1]])
        
        midline = pair_cones_to_midline(left, right, method='hungarian')
        
        assert len(midline) == 3
        # Should produce midline close to y=0
        assert np.all(np.abs(midline[:, 1]) < 0.1)
    
    def test_hungarian_unequal_cones(self):
        """Test Hungarian pairing with unequal cone numbers."""
        left = np.array([[0, 1], [1, 1]])
        right = np.array([[0, -1], [1, -1], [2, -1]])
        
        midline = pair_cones_to_midline(left, right, method='hungarian')
        
        assert len(midline) == 2  # Should match smaller set
    
    def test_hungarian_vs_greedy(self):
        """Compare Hungarian vs greedy pairing."""
        # Create scenario where greedy might fail
        left = np.array([[0, 1], [2, 1]])
        right = np.array([[1, -1], [0, -1]])
        
        midline_greedy = pair_cones_to_midline(left, right, method='greedy')
        midline_hungarian = pair_cones_to_midline(left, right, method='hungarian')
        
        assert len(midline_greedy) == len(midline_hungarian)
        # Hungarian should potentially give better pairing


class TestSplineSmoothing:
    """Test spline-based smoothing."""
    
    def test_spline_smooth_basic(self):
        """Test basic spline smoothing."""
        path = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        
        smoothed = spline_smooth(path)
        
        assert len(smoothed) == len(path)
        assert smoothed.shape[1] == 2
    
    def test_spline_with_corridor(self):
        """Test spline smoothing with corridor constraints."""
        path = np.array([[0, 0], [1, 0], [2, 0]])
        left_cones = np.array([[0, 1], [1, 1], [2, 1]])
        right_cones = np.array([[0, -1], [1, -1], [2, -1]])
        
        smoothed = spline_smooth(path, corridor=(left_cones, right_cones))
        
        assert len(smoothed) == len(path)
        # Should stay within corridor
        assert np.all(smoothed[:, 1] >= -1.1)
        assert np.all(smoothed[:, 1] <= 1.1)
    
    def test_spline_different_num_points(self):
        """Test spline with different output point count."""
        path = np.array([[0, 0], [1, 1], [2, 0]])
        
        smoothed = spline_smooth(path, num_points=10)
        
        assert len(smoothed) == 10
    
    def test_spline_short_path(self):
        """Test spline with very short path."""
        path = np.array([[0, 0], [1, 1]])
        
        smoothed = spline_smooth(path)
        
        # Should return original path for short paths
        assert len(smoothed) == len(path)


class TestOptimizationRacingLine:
    """Test optimization-based racing line generation."""
    
    def test_optimization_basic(self):
        """Test basic optimization racing line."""
        left = np.array([[0, 1], [1, 1], [2, 1]])
        right = np.array([[0, -1], [1, -1], [2, -1]])
        
        racing_line = optimization_based_racing_line(left, right)
        
        assert len(racing_line) > 0
        assert racing_line.shape[1] == 2
    
    def test_optimization_parameters(self):
        """Test optimization with different parameters."""
        left = np.array([[0, 1], [1, 1], [2, 1]])
        right = np.array([[0, -1], [1, -1], [2, -1]])
        
        line1 = optimization_based_racing_line(
            left, right, curvature_weight=1.0, smoothness_weight=0.1)
        line2 = optimization_based_racing_line(
            left, right, curvature_weight=0.1, smoothness_weight=1.0)
        
        assert len(line1) == len(line2)
        # Different weights should produce different lines
    
    def test_optimization_insufficient_cones(self):
        """Test optimization with insufficient cone data."""
        left = np.array([[0, 1]])
        right = np.array([[0, -1]])
        
        racing_line = optimization_based_racing_line(left, right)
        
        # Should fallback to standard pairing
        assert len(racing_line) > 0


class TestCorridorConstraints:
    """Test corridor constraint application."""
    
    def test_apply_corridor_constraints(self):
        """Test corridor constraint application."""
        path = np.array([[0, 2], [1, -2], [2, 0]])  # Path outside corridor
        left_cones = np.array([[0, 1], [1, 1], [2, 1]])
        right_cones = np.array([[0, -1], [1, -1], [2, -1]])
        
        constrained = _apply_corridor_constraints(path, (left_cones, right_cones))
        
        # Should be pulled back into corridor
        assert np.all(constrained[:, 1] <= 1.1)
        assert np.all(constrained[:, 1] >= -1.1)
    
    def test_corridor_constraints_inside(self):
        """Test corridor constraints with path already inside."""
        path = np.array([[0, 0], [1, 0], [2, 0]])  # Path inside corridor
        left_cones = np.array([[0, 1], [1, 1], [2, 1]])
        right_cones = np.array([[0, -1], [1, -1], [2, -1]])
        
        constrained = _apply_corridor_constraints(path, (left_cones, right_cones))
        
        # Should remain largely unchanged
        np.testing.assert_array_almost_equal(path, constrained, decimal=1)


class TestBoundaryInterpolation:
    """Test boundary interpolation functionality."""
    
    def test_interpolate_boundary_basic(self):
        """Test basic boundary interpolation."""
        cones = np.array([[0, 0], [1, 1], [2, 0]])
        t_param = np.linspace(0, 1, 5)
        
        interpolated = _interpolate_boundary(cones, t_param)
        
        assert len(interpolated) == 5
        assert interpolated.shape[1] == 2
    
    def test_interpolate_single_cone(self):
        """Test interpolation with single cone."""
        cones = np.array([[1, 2]])
        t_param = np.linspace(0, 1, 3)
        
        interpolated = _interpolate_boundary(cones, t_param)
        
        # Should tile the single cone
        assert len(interpolated) == 3
        np.testing.assert_array_equal(interpolated, np.tile([1, 2], (3, 1)))
    
    def test_interpolate_empty_cones(self):
        """Test interpolation with no cones."""
        cones = np.array([]).reshape(0, 2)
        t_param = np.linspace(0, 1, 3)
        
        interpolated = _interpolate_boundary(cones, t_param)
        
        # Should return zeros
        assert len(interpolated) == 3
        np.testing.assert_array_equal(interpolated, np.zeros((3, 2)))


class TestGeometricCurvature:
    """Test geometric curvature calculation."""
    
    def test_curvature_straight_line(self):
        """Test curvature of straight line."""
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        
        curvature = curvature_geometric(path)
        
        # Straight line should have zero curvature
        assert np.all(curvature < 0.01)
    
    def test_curvature_circle(self):
        """Test curvature of circular path."""
        # Create circular path
        theta = np.linspace(0, np.pi, 20)
        radius = 2.0
        path = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        
        curvature = curvature_geometric(path)
        
        # Circle should have constant curvature = 1/radius
        expected_curvature = 1.0 / radius
        # Allow some tolerance due to discrete approximation
        assert np.all(np.abs(curvature - expected_curvature) < 0.2)
    
    def test_curvature_short_path(self):
        """Test curvature with short path."""
        path = np.array([[0, 0], [1, 1]])
        
        curvature = curvature_geometric(path)
        
        assert len(curvature) == 2
        assert np.all(curvature == 0)  # Should be zero for short paths
    
    def test_curvature_vs_discrete(self):
        """Compare geometric vs discrete curvature."""
        # Create path with known curvature
        t = np.linspace(0, 2*np.pi, 50)
        path = np.column_stack([np.cos(t), np.sin(t)])  # Unit circle
        
        curv_geometric = curvature_geometric(path)
        curv_discrete = curvature_discrete(path)
        
        # Both should detect curvature, geometric should be more accurate
        assert np.mean(curv_geometric) > 0.5  # Should be close to 1.0
        assert np.mean(curv_discrete) > 0.1   # Should also detect curvature
