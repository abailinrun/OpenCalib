/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 * Ouyang Jinhua <ouyangjinhua@pjlab.org.cn>
 *
 * Modified for MEMS LiDAR support (ring-free calibration methods)
 * by abailinrun, 2024
 */
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include "calibration.hpp"
#include "calibration_ringfree.hpp"
#include "extrinsic_param.hpp"
#include "registration.hpp"
#include <iostream>
#include <cstring>

#include "logging.hpp"
#include "transform_util.hpp"
using namespace std;

void printUsage() {
  cout << "Usage: ./run_lidar2imu <lidar_pcds_dir> <poses_path> "
          "<extrinsic_json> [--method <balm|pca|voxel|ground>]"
          "\n\nMethods:"
          "\n  balm   : LOAM-style feature extraction with BALM optimization (default)"
          "\n           Requires valid ring field in PCD (mechanical LiDAR)"
          "\n  pca    : PCA-based feature extraction (ring-free, RECOMMENDED for MEMS)"
          "\n           Uses K-NN + covariance analysis for feature classification"
          "\n           Full 6-DOF optimization, large search range (20°+)"
          "\n  voxel  : Voxel occupancy based optimization (ring-free, refinement only)"
          "\n           Limited search range (±3° yaw, ±0.6m tx/ty)"
          "\n           Only optimizes yaw, tx, ty (not roll, pitch, tz)"
          "\n  ground : Ground plane based optimization (ring-free, refinement only)"
          "\n           Limited search range (±6° roll/pitch)"
          "\n           Only optimizes roll, pitch (not yaw, tx, ty, tz)"
          "\n\nExamples:"
          "\n  # Standard calibration (mechanical LiDAR with valid ring field)"
          "\n  ./bin/run_lidar2imu data/top_center_lidar/ \\"
          "\n      data/NovAtel-pose-lidar-time.txt \\"
          "\n      data/gnss-to-top_center_lidar-extrinsic.json"
          "\n"
          "\n  # MEMS LiDAR calibration (ring-free, PCA method - RECOMMENDED)"
          "\n  ./bin/run_lidar2imu data/m1_lidar/ \\"
          "\n      data/NovAtel-pose-lidar-time.txt \\"
          "\n      data/gnss-to-sensor_kit_base_link-extrinsic.json \\"
          "\n      --method pca"
          "\n"
          "\n  # Refinement only (requires good initial extrinsic < 3°)"
          "\n  ./bin/run_lidar2imu data/m1_lidar/ \\"
          "\n      data/NovAtel-pose-lidar-time.txt \\"
          "\n      data/gnss-to-sensor_kit_base_link-extrinsic.json \\"
          "\n      --method voxel"
          "\n"
       << endl;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printUsage();
    return 0;
  }

  string lidar_pcds_dir = argv[1];
  string poses_path = argv[2];
  string extrinsic_json = argv[3];
  string method = "balm";  // Default is BALM (original method)
  string stitching_path = "stitching.pcd";

  // Parse command line arguments
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
      method = argv[i + 1];
      i++;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printUsage();
      return 0;
    }
  }

  // Validate method
  if (method != "balm" && method != "pca" && method != "voxel" && method != "ground") {
    cerr << "ERROR: Unknown method '" << method << "'" << endl;
    cerr << "Valid methods: balm, pca, voxel, ground" << endl;
    return 1;
  }

  // Load extrinsic parameters
  Eigen::Matrix4d json_param;
  LoadExtrinsic(extrinsic_json, json_param);
  LOGI("Load extrinsic!");

  // Convert to lidar2imu
  Eigen::Matrix4d lidar2imu_extrinsic = json_param.inverse().eval();
  std::cout << "Initial extrinsic (GNSS to LiDAR):" << std::endl;
  std::cout << json_param << std::endl;

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

  if (method == "balm") {
    // Original BALM method (requires valid ring field)
    LOGI("Using BALM method (requires valid ring field)");
    Calibrator calibrator;
    calibrator.Calibration(lidar_pcds_dir, poses_path, lidar2imu_extrinsic);
  } else if (method == "pca") {
    // PCA-based method (ring-free, MEMS LiDAR compatible)
    LOGI("Using PCA-based method (ring-free, MEMS LiDAR compatible)");
    LOGI("This method uses K-NN + covariance analysis for feature extraction");
    LOGI("Full 6-DOF optimization with large search range (20°+)");
    RingFreeCalibrator calibrator;
    calibrator.Calibration(lidar_pcds_dir, poses_path, lidar2imu_extrinsic);
  } else if (method == "voxel") {
    // Voxel occupancy method (ring-free, MEMS LiDAR compatible)
    LOGI("Using Voxel Occupancy method (ring-free, MEMS LiDAR compatible)");
    Registrator registrator;
    registrator.LoadOdometerData(poses_path, lidar2imu_extrinsic);
    registrator.LoadLidarPCDs(lidar_pcds_dir);
    registrator.RegistrationByVoxelOccupancy(transform);
    registrator.SaveStitching(stitching_path);

    std::cout << "\n=== Calibration Result (IMU to LiDAR) ===" << std::endl;
    std::cout << transform << std::endl;

    // Save result to file
    std::string result_file = "./refined_calib_imu_to_lidar.txt";
    std::ofstream fout(result_file);
    if (fout.is_open()) {
      fout << "refined calib (voxel method):" << std::endl;
      fout << "R: " << transform(0, 0) << " " << transform(0, 1) << " "
           << transform(0, 2) << " " << transform(1, 0) << " "
           << transform(1, 1) << " " << transform(1, 2) << " "
           << transform(2, 0) << " " << transform(2, 1) << " "
           << transform(2, 2) << std::endl;
      fout << "t: " << transform(0, 3) << " " << transform(1, 3) << " "
           << transform(2, 3) << std::endl;
      fout.close();
      LOGI("Saved result to %s", result_file.c_str());
    }
  } else if (method == "ground") {
    // Ground plane method (ring-free, optimizes roll/pitch)
    LOGI("Using Ground Plane method (ring-free, optimizes roll/pitch)");
    Registrator registrator;
    registrator.LoadOdometerData(poses_path, lidar2imu_extrinsic);
    registrator.LoadLidarPCDs(lidar_pcds_dir);
    registrator.RegistrationByGroundPlane(transform);
    registrator.SaveStitching(stitching_path);

    std::cout << "\n=== Calibration Result (IMU to LiDAR) ===" << std::endl;
    std::cout << transform << std::endl;

    // Save result to file
    std::string result_file = "./refined_calib_imu_to_lidar.txt";
    std::ofstream fout(result_file);
    if (fout.is_open()) {
      fout << "refined calib (ground plane method):" << std::endl;
      fout << "R: " << transform(0, 0) << " " << transform(0, 1) << " "
           << transform(0, 2) << " " << transform(1, 0) << " "
           << transform(1, 1) << " " << transform(1, 2) << " "
           << transform(2, 0) << " " << transform(2, 1) << " "
           << transform(2, 2) << std::endl;
      fout << "t: " << transform(0, 3) << " " << transform(1, 3) << " "
           << transform(2, 3) << std::endl;
      fout.close();
      LOGI("Saved result to %s", result_file.c_str());
    }
  }

  return 0;
}