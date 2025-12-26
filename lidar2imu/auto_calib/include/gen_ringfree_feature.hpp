/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 *
 * Ring-free feature extraction for MEMS LiDAR support
 * Modified by abailinrun, 2024
 *
 * This implementation replaces the ring-dependent LOAM-style feature extraction
 * with a PCA-based method that works with any LiDAR (including MEMS like RoboSense M1).
 *
 * Key differences from gen_BALM_feature.hpp:
 * - No ring field dependency
 * - Uses KD-tree for K-NN neighborhood search
 * - Computes local covariance matrix and PCA for feature classification
 * - Edge features: lambda1 >> lambda2 (linear structure)
 * - Planar features: lambda3 << lambda1, lambda2 (planar structure)
 */

#ifndef GEN_RINGFREE_FEATURE_HPP
#define GEN_RINGFREE_FEATURE_HPP

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common/Lidar_parser_base.h"

// Configuration parameters for PCA-based feature extraction
namespace RingFreeConfig {
  // K-NN neighborhood size for PCA computation
  // Larger K: more robust but slower, may over-smooth features
  // Smaller K: faster but more sensitive to noise
  constexpr int KNN_NEIGHBORS = 15;

  // Minimum number of neighbors required for valid PCA
  constexpr int MIN_NEIGHBORS = 5;

  // Eigenvalue ratio thresholds for feature classification
  // Planar feature: smallest eigenvalue is much smaller than others
  constexpr float PLANAR_THRESHOLD = 0.1f;  // lambda3 / lambda1 < threshold

  // Edge feature: largest eigenvalue dominates
  constexpr float EDGE_THRESHOLD = 0.7f;  // (lambda1 - lambda2) / lambda1 > threshold

  // Intensity threshold (same as original)
  constexpr float INTENSITY_THRESHOLD = 35.0f;

  // Maximum points per feature type per segment
  constexpr int MAX_EDGE_PER_SEGMENT = 2;
  constexpr int MAX_PLANAR_PER_SEGMENT = 4;

  // Number of angular segments for feature distribution
  constexpr int NUM_SEGMENTS = 6;

  // Voxel size for downsampling planar features
  constexpr float DOWNSAMPLE_LEAF_SIZE = 0.2f;

  // Distance threshold for neighbor picking (prevent selecting occluded points)
  constexpr float NEIGHBOR_DISTANCE_THRESHOLD = 0.05f;
}

/**
 * Structure to hold PCA analysis results for a point
 */
struct PCAResult {
  Eigen::Vector3f eigenvalues;   // Sorted: lambda1 >= lambda2 >= lambda3
  Eigen::Matrix3f eigenvectors;  // Corresponding eigenvectors
  Eigen::Vector3f normal;        // Normal vector (for planar features)
  float planarity;               // Planarity score: (lambda2 - lambda3) / lambda1
  float linearity;               // Linearity score: (lambda1 - lambda2) / lambda1
  bool is_valid;                 // Whether PCA computation succeeded
};

/**
 * Compute PCA for a point's local neighborhood
 *
 * @param cloud Input point cloud
 * @param kdtree Pre-built KD-tree for the cloud
 * @param point_idx Index of the query point
 * @param K Number of neighbors to use
 * @return PCAResult containing eigenvalues, eigenvectors, and feature scores
 */
inline PCAResult computeLocalPCA(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const pcl::KdTreeFLANN<pcl::PointXYZI>& kdtree,
    int point_idx,
    int K = RingFreeConfig::KNN_NEIGHBORS) {

  PCAResult result;
  result.is_valid = false;

  // K-NN search
  std::vector<int> neighbor_indices(K);
  std::vector<float> neighbor_distances(K);

  const pcl::PointXYZI& query_point = cloud->points[point_idx];

  int found = kdtree.nearestKSearch(query_point, K, neighbor_indices, neighbor_distances);

  if (found < RingFreeConfig::MIN_NEIGHBORS) {
    return result;
  }

  // Compute centroid
  Eigen::Vector3f centroid(0, 0, 0);
  for (int i = 0; i < found; ++i) {
    const pcl::PointXYZI& p = cloud->points[neighbor_indices[i]];
    centroid += Eigen::Vector3f(p.x, p.y, p.z);
  }
  centroid /= static_cast<float>(found);

  // Compute covariance matrix
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (int i = 0; i < found; ++i) {
    const pcl::PointXYZI& p = cloud->points[neighbor_indices[i]];
    Eigen::Vector3f diff(p.x - centroid.x(), p.y - centroid.y(), p.z - centroid.z());
    covariance += diff * diff.transpose();
  }
  covariance /= static_cast<float>(found);

  // Eigen decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return result;
  }

  // Eigenvalues are sorted in ascending order by Eigen
  // We want descending order: lambda1 >= lambda2 >= lambda3
  Eigen::Vector3f eigenvalues = solver.eigenvalues().reverse();
  Eigen::Matrix3f eigenvectors = solver.eigenvectors().rowwise().reverse();

  // Ensure non-negative eigenvalues
  for (int i = 0; i < 3; ++i) {
    if (eigenvalues[i] < 0) eigenvalues[i] = 0;
  }

  result.eigenvalues = eigenvalues;
  result.eigenvectors = eigenvectors;

  // Normal vector is the eigenvector corresponding to smallest eigenvalue
  result.normal = eigenvectors.col(2);

  // Compute feature scores
  float sum_eigenvalues = eigenvalues.sum();
  if (sum_eigenvalues > 1e-6f) {
    result.planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0];
    result.linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
  } else {
    result.planarity = 0;
    result.linearity = 0;
  }

  result.is_valid = true;
  return result;
}

/**
 * Ring-free feature extraction using PCA-based analysis
 *
 * This function replaces genPcdFeature() from gen_BALM_feature.hpp
 * and does not require the ring field.
 *
 * @param laserCloud Input point cloud (can be LidarPointXYZIRT or PointXYZI)
 * @param pcd_surf Output: planar surface features (less sharp)
 * @param pcd_surf_sharp Output: sharp planar features
 * @param pcd_corn Output: corner/edge features
 * @return true if successful, false otherwise
 */
template<typename PointT>
bool genPcdFeatureRingFree(
    const typename pcl::PointCloud<PointT>::Ptr& laserCloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_surf,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_surf_sharp,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_corn) {

  int cloud_size = laserCloud->points.size();
  if (cloud_size < RingFreeConfig::KNN_NEIGHBORS * 2) {
    std::cerr << "[ERROR] Point cloud too small for feature extraction: "
              << cloud_size << " points" << std::endl;
    return false;
  }

  std::cout << "[RingFree] Processing " << cloud_size << " points..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();

  // Convert to PointXYZI for KD-tree (if needed)
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
  cloud_xyzi->reserve(cloud_size);

  for (int i = 0; i < cloud_size; ++i) {
    const PointT& src = laserCloud->points[i];
    pcl::PointXYZI dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.intensity = src.intensity;
    cloud_xyzi->push_back(dst);
  }

  // Build KD-tree
  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
  kdtree.setInputCloud(cloud_xyzi);

  // Temporary storage for features with their scores
  struct FeatureCandidate {
    int index;
    float score;
    pcl::PointXYZI point;
  };

  // Compute azimuth angle for each point (for angular segmentation)
  std::vector<float> azimuth_angles(cloud_size);
  for (int i = 0; i < cloud_size; ++i) {
    const pcl::PointXYZI& p = cloud_xyzi->points[i];
    azimuth_angles[i] = std::atan2(p.y, p.x);
  }

  // Feature containers per segment
  std::vector<std::vector<FeatureCandidate>> edge_candidates(RingFreeConfig::NUM_SEGMENTS);
  std::vector<std::vector<FeatureCandidate>> planar_candidates(RingFreeConfig::NUM_SEGMENTS);

  // Process each point
  int valid_pca_count = 0;
  int edge_count = 0;
  int planar_count = 0;

  #pragma omp parallel for reduction(+:valid_pca_count, edge_count, planar_count)
  for (int i = 0; i < cloud_size; ++i) {
    const pcl::PointXYZI& point = cloud_xyzi->points[i];

    // Skip low intensity points
    if (point.intensity < RingFreeConfig::INTENSITY_THRESHOLD) {
      continue;
    }

    // Skip invalid points
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }

    // Compute PCA for local neighborhood
    PCAResult pca = computeLocalPCA(cloud_xyzi, kdtree, i);
    if (!pca.is_valid) {
      continue;
    }

    valid_pca_count++;

    // Determine angular segment
    float azimuth = azimuth_angles[i];
    int segment = static_cast<int>((azimuth + M_PI) / (2 * M_PI) * RingFreeConfig::NUM_SEGMENTS);
    segment = std::min(std::max(segment, 0), RingFreeConfig::NUM_SEGMENTS - 1);

    // Classify feature type based on eigenvalue ratios
    float lambda_ratio = pca.eigenvalues[2] / (pca.eigenvalues[0] + 1e-6f);

    FeatureCandidate candidate;
    candidate.index = i;
    candidate.point = point;

    // Check for edge feature (high linearity)
    if (pca.linearity > RingFreeConfig::EDGE_THRESHOLD) {
      candidate.score = pca.linearity;
      #pragma omp critical
      {
        edge_candidates[segment].push_back(candidate);
        edge_count++;
      }
    }
    // Check for planar feature (low lambda3/lambda1 ratio)
    else if (lambda_ratio < RingFreeConfig::PLANAR_THRESHOLD) {
      candidate.score = 1.0f - lambda_ratio;  // Higher score for more planar
      #pragma omp critical
      {
        planar_candidates[segment].push_back(candidate);
        planar_count++;
      }
    }
  }

  // Select best features from each segment
  pcl::PointCloud<pcl::PointXYZI> cornerPointsSharp;
  pcl::PointCloud<pcl::PointXYZI> surfPointsFlat;
  pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>());

  for (int seg = 0; seg < RingFreeConfig::NUM_SEGMENTS; ++seg) {
    // Sort edge candidates by score (descending)
    auto& edges = edge_candidates[seg];
    std::sort(edges.begin(), edges.end(),
              [](const FeatureCandidate& a, const FeatureCandidate& b) {
                return a.score > b.score;
              });

    // Select top edge features
    int edge_selected = 0;
    for (const auto& candidate : edges) {
      if (edge_selected >= RingFreeConfig::MAX_EDGE_PER_SEGMENT) break;
      cornerPointsSharp.push_back(candidate.point);
      edge_selected++;
    }

    // Sort planar candidates by score (descending)
    auto& planars = planar_candidates[seg];
    std::sort(planars.begin(), planars.end(),
              [](const FeatureCandidate& a, const FeatureCandidate& b) {
                return a.score > b.score;
              });

    // Select top planar features (sharp)
    int planar_selected = 0;
    for (const auto& candidate : planars) {
      if (planar_selected >= RingFreeConfig::MAX_PLANAR_PER_SEGMENT) break;
      surfPointsFlat.push_back(candidate.point);
      planar_selected++;
    }

    // All planar candidates go to less flat (will be downsampled)
    for (const auto& candidate : planars) {
      surfPointsLessFlat->push_back(candidate.point);
    }
  }

  // Downsample less flat surface points
  pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlatDS;
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
  downSizeFilter.setInputCloud(surfPointsLessFlat);
  downSizeFilter.setLeafSize(RingFreeConfig::DOWNSAMPLE_LEAF_SIZE,
                             RingFreeConfig::DOWNSAMPLE_LEAF_SIZE,
                             RingFreeConfig::DOWNSAMPLE_LEAF_SIZE);
  downSizeFilter.filter(surfPointsLessFlatDS);

  // Copy to output
  *pcd_corn = cornerPointsSharp;
  *pcd_surf_sharp = surfPointsFlat;
  *pcd_surf = surfPointsLessFlatDS;

  auto end_time = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  std::cout << "[RingFree] Feature extraction completed in " << elapsed << "s" << std::endl;
  std::cout << "[RingFree] Valid PCA computations: " << valid_pca_count << std::endl;
  std::cout << "[RingFree] Edge candidates: " << edge_count
            << ", selected: " << cornerPointsSharp.size() << std::endl;
  std::cout << "[RingFree] Planar candidates: " << planar_count
            << ", sharp: " << surfPointsFlat.size()
            << ", less flat: " << surfPointsLessFlatDS.size() << std::endl;

  // Validation check
  if (cornerPointsSharp.size() < 10 || surfPointsFlat.size() < 10) {
    std::cerr << "[WARNING] Very few features extracted. "
              << "Consider adjusting thresholds or checking point cloud quality." << std::endl;
  }

  return true;
}

/**
 * Wrapper function matching the original genPcdFeature signature
 * for drop-in replacement compatibility
 */
inline bool genPcdFeature_RingFree(
    pcl::PointCloud<LidarPointXYZIRT>::Ptr laserCloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_surf,
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_surf_sharp,
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_corn) {

  return genPcdFeatureRingFree<LidarPointXYZIRT>(laserCloud, pcd_surf, pcd_surf_sharp, pcd_corn);
}

#endif // GEN_RINGFREE_FEATURE_HPP
