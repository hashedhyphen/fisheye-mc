#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>

class MotionEstimator
{
public:
  static const int kSearchRange = 32;

  MotionEstimator(
    const cv::Mat1b& aCurrentFrame,
    const cv::Mat1b& aReferenceFrame,
    int aBlockSize
  );

  virtual ~MotionEstimator() = default;

  MotionEstimator(const MotionEstimator&) = default;
  MotionEstimator& operator=(const MotionEstimator&) = default;

  MotionEstimator(MotionEstimator&&) = default;
  MotionEstimator& operator=(MotionEstimator&&) = default;

  int blockSize() const { return mBlockSize; }

  cv::Mat1b generatePredictedFrame();

  void dumpSummary() const;

  cv::Mat1b visualizePredictionErrors() const;

  cv::Mat3b visualizeMotionIndices() const;

protected:
  static const std::vector<cv::Vec2i> kSpiralOffsets;

  bool isOutOfFrame(int x, int y) const
  {
    return x < 0 || mReferenceFrame.cols <= x ||
           y < 0 || mReferenceFrame.rows <= y;
  }

  bool isOutOfInterpolatedFrame(int x, int y) const
  {
    return x < 0 || mInterpolatedFrame.cols <= x ||
           y < 0 || mInterpolatedFrame.rows <= y;
  }

  const cv::Mat1b mCurrentFrame;
  const cv::Mat1b mReferenceFrame;
  const cv::Mat1b mInterpolatedFrame;
  cv::Mat1b mPredictedFrame;

  const int mBlockSize;

  cv::Mat2i mMotionParamsIndicesMap;

private:
  using Distortion = std::uint32_t;

  static const std::array<int, 6> k6TapFilter;

  static cv::Mat1b interpolateReference(
    const cv::Mat1b& aReference
  );

  static cv::Mat1b createCircleMask(
    int aFrameSize,
    int aBlockSize
  );

  static cv::Mat1b createIndicesCircleMask(
    int aFrameSize,
    int aBlockSize
  );

  cv::Vec2i estimateMotionParamsIndices(
    int i,
    int j
  ) const;

  virtual cv::Mat1b calcReferenceBlock(
    int i,
    int j,
    const cv::Vec2i& aMotionParamsIndices
  ) const = 0;

  void dumpPSNR() const;

  bool isCodingBlock(
    int i,
    int j
  ) const
  {
    return 0 <= i && i < mMotionParamsIndicesMap.cols
        && 0 <= j && j < mMotionParamsIndicesMap.rows
        && mIndicesCircleMask.at<std::uint8_t>(j, i) > 0;
  }

  cv::Vec2i deriveMedianPredictor(
    int i,
    int j
  ) const;

  void dumpEntropies() const;

  const cv::Mat1b mCircleMask;
  const cv::Mat1b mIndicesCircleMask;
};
