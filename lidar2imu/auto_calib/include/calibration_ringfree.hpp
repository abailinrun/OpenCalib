/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 *
 * Ring-free calibration for MEMS LiDAR support
 * Modified by abailinrun, 2024
 *
 * This class provides LiDAR-IMU calibration that does not depend on the ring field,
 * making it compatible with MEMS LiDAR sensors like RoboSense M1.
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/Lidar_parser_base.h"

/**
 * Ring-free LiDAR-IMU Calibrator
 *
 * Uses PCA-based feature extraction instead of ring-dependent LOAM features.
 * Compatible with both mechanical LiDAR and MEMS LiDAR.
 */
class RingFreeCalibrator {
public:
  RingFreeCalibrator();
  ~RingFreeCalibrator();

  /**
   * Load timestamp and pose data from odometer file
   *
   * @param filename Path to pose file (format: timestamp + 3x4 transformation matrix)
   * @param Tl2i Initial LiDAR to IMU transformation
   * @param lidarTimes Output: list of timestamps
   * @param lidarPoses Output: list of poses
   */
  void LoadTimeAndPoes(const std::string &filename, const Eigen::Matrix4d &Tl2i,
                       std::vector<std::string> &lidarTimes,
                       std::vector<Eigen::Matrix4d> &lidarPoses);

  /**
   * Convert rotation (angle-axis) and translation to transformation matrix
   */
  Eigen::Matrix4d GetDeltaTrans(double R[3], double t[3]);

  /**
   * Run the ring-free calibration
   *
   * @param lidar_path Directory containing LiDAR PCD files
   * @param odom_path Path to odometer/pose file
   * @param init_Tl2i Initial LiDAR to IMU transformation (from CAD or manual measurement)
   */
  void Calibration(const std::string lidar_path, const std::string odom_path,
                   const Eigen::Matrix4d init_Tl2i);

  /**
   * Save stitched point cloud for visualization
   */
  void SaveStitching(const Eigen::Matrix4d transform,
                     const std::string pcd_name);

public:
  // Calibration parameters (public for configuration)
  int turn_ = 20;      // Number of optimization rounds
  int window_ = 10;    // Sliding window size

private:

  // Data storage
  std::vector<std::string> lidar_files_;
  std::vector<Eigen::Matrix4d> lidar_poses_;

  // Constants
  double degree_2_radian = 0.017453293;
  std::string lidar_path_;
};
