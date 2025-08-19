"""ROS 2 interface for the racing line optimizer.

This module provides ROS 2 topic interfaces for:
- Cone detection input
- Path planning output  
- Vehicle state telemetry
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point, PoseStamped, Twist
    from nav_msgs.msg import Path
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header, Float64MultiArray
    from visualization_msgs.msg import MarkerArray, Marker
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Create dummy classes for when ROS 2 is not available
    class Node:
        pass

from .planner import pair_cones_to_midline, laplacian_smooth, spline_smooth, optimization_based_racing_line
from .models import VehicleParams, BicycleKinematic, pure_pursuit_control, stanley_control, TelemetryLogger


class RacingLineOptimizerNode(Node):
    """ROS 2 node for racing line optimization."""
    
    def __init__(self):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS 2 not available. Install rclpy and related packages.")
            
        super().__init__('racing_line_optimizer')
        
        # Parameters
        self.declare_parameter('pairing_method', 'hungarian')
        self.declare_parameter('smoothing_method', 'laplacian')
        self.declare_parameter('controller_type', 'pure_pursuit')
        self.declare_parameter('publish_rate', 10.0)
        
        # Vehicle setup
        self.vehicle_params = VehicleParams()
        self.vehicle_model = BicycleKinematic(self.vehicle_params)
        self.telemetry_logger = TelemetryLogger()
        
        # State
        self.left_cones = np.array([])
        self.right_cones = np.array([])
        self.current_path = np.array([])
        self.vehicle_state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, yaw, v
        
        # Subscribers
        self.cone_sub = self.create_subscription(
            Float64MultiArray, '/cones/left_right', self.cone_callback, 10)
        self.vehicle_state_sub = self.create_subscription(
            PoseStamped, '/vehicle/pose', self.vehicle_state_callback, 10)
        self.velocity_sub = self.create_subscription(
            Twist, '/vehicle/velocity', self.velocity_callback, 10)
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/racing_line/path', 10)
        self.control_pub = self.create_publisher(Twist, '/vehicle/cmd_vel', 10)
        self.telemetry_pub = self.create_publisher(
            Float64MultiArray, '/racing_line/telemetry', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/racing_line/visualization', 10)
        
        # Timer for control loop
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.control_loop)
        
        self.get_logger().info('Racing Line Optimizer Node initialized')
    
    def cone_callback(self, msg):
        """Process cone detection data."""
        try:
            # Expected format: [left_x1, left_y1, left_x2, left_y2, ..., 
            #                   right_x1, right_y1, right_x2, right_y2, ...]
            data = np.array(msg.data)
            
            if len(data) % 4 != 0:
                self.get_logger().warn('Invalid cone data format')
                return
            
            # Split into left and right cones
            mid_point = len(data) // 2
            left_data = data[:mid_point].reshape(-1, 2)
            right_data = data[mid_point:].reshape(-1, 2)
            
            self.left_cones = left_data
            self.right_cones = right_data
            
            # Generate new path
            self._update_racing_line()
            
        except Exception as e:
            self.get_logger().error(f'Error processing cone data: {e}')
    
    def vehicle_state_callback(self, msg):
        """Update vehicle position and orientation."""
        self.vehicle_state[0] = msg.pose.position.x
        self.vehicle_state[1] = msg.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.vehicle_state[2] = yaw
    
    def velocity_callback(self, msg):
        """Update vehicle velocity."""
        self.vehicle_state[3] = msg.linear.x
    
    def _update_racing_line(self):
        """Generate new racing line from cone data."""
        if len(self.left_cones) == 0 or len(self.right_cones) == 0:
            return
        
        try:
            pairing_method = self.get_parameter('pairing_method').value
            smoothing_method = self.get_parameter('smoothing_method').value
            
            # Generate racing line based on parameters
            if smoothing_method == 'optimization':
                self.current_path = optimization_based_racing_line(
                    self.left_cones, self.right_cones)
            else:
                # Standard approach
                midline = pair_cones_to_midline(
                    self.left_cones, self.right_cones, method=pairing_method)
                
                if smoothing_method == 'spline':
                    self.current_path = spline_smooth(
                        midline, corridor=(self.left_cones, self.right_cones))
                else:
                    self.current_path = laplacian_smooth(
                        midline, corridor=(self.left_cones, self.right_cones))
            
            # Publish path
            self._publish_path()
            self._publish_visualization()
            
        except Exception as e:
            self.get_logger().error(f'Error updating racing line: {e}')
    
    def control_loop(self):
        """Main control loop."""
        if len(self.current_path) < 2:
            return
        
        try:
            controller_type = self.get_parameter('controller_type').value
            
            # Compute control command
            if controller_type == 'stanley':
                steer, _ = stanley_control(
                    self.vehicle_state, self.current_path,
                    wheelbase=self.vehicle_params.wheelbase)
            else:  # pure_pursuit
                steer, _ = pure_pursuit_control(
                    self.vehicle_state, self.current_path)
            
            # Simple speed control (maintain target speed)
            target_speed = 5.0  # m/s
            speed_error = target_speed - self.vehicle_state[3]
            accel = 2.0 * speed_error  # Proportional control
            
            # Publish control command
            cmd_msg = Twist()
            cmd_msg.linear.x = target_speed
            cmd_msg.angular.z = steer
            self.control_pub.publish(cmd_msg)
            
            # Log telemetry
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.telemetry_logger.log(
                current_time, self.vehicle_state, (steer, accel), self.current_path)
            
            # Publish telemetry
            self._publish_telemetry()
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
    
    def _publish_path(self):
        """Publish racing line as ROS Path message."""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for point in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
    
    def _publish_telemetry(self):
        """Publish telemetry data."""
        if len(self.telemetry_logger.data['time']) == 0:
            return
        
        # Get latest telemetry
        latest_data = []
        for key in ['x', 'y', 'yaw', 'velocity', 'steering', 'acceleration',
                   'cross_track_error', 'heading_error']:
            if self.telemetry_logger.data[key]:
                latest_data.append(self.telemetry_logger.data[key][-1])
            else:
                latest_data.append(0.0)
        
        telemetry_msg = Float64MultiArray()
        telemetry_msg.data = latest_data
        self.telemetry_pub.publish(telemetry_msg)
    
    def _publish_visualization(self):
        """Publish visualization markers."""
        marker_array = MarkerArray()
        
        # Racing line marker
        if len(self.current_path) > 0:
            line_marker = Marker()
            line_marker.header.frame_id = 'map'
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.1
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0
            
            for point in self.current_path:
                p = Point()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = 0.0
                line_marker.points.append(p)
            
            marker_array.markers.append(line_marker)
        
        # Cone markers
        for i, cone in enumerate(self.left_cones):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i + 100
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(cone[0])
            marker.pose.position.y = float(cone[1])
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        for i, cone in enumerate(self.right_cones):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i + 200
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(cone[0])
            marker.pose.position.y = float(cone[1])
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    """Main entry point for ROS 2 node."""
    if not ROS2_AVAILABLE:
        print("ROS 2 not available. Please install rclpy and related packages.")
        return
    
    rclpy.init(args=args)
    node = RacingLineOptimizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
