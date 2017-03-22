#pragma once

#include "motion_estimator.hpp"

class Radial : public MotionEstimator
{
public:
  Radial(
    const cv::Mat1b& aCurrentFrame,
    const cv::Mat1b& aReferenceFrame,
    const int aBlockSize
  );

private:
  enum class BlockType : std::uint8_t {
    eNone,
    eTrans,
    eDual
  };

  static const double kSearchStepDr;
  static const double kSearchStepDl;

  static cv::Mat1b createBlockTypeMap(
    int aFrameSize,
    int aBlockSize
  );

  cv::Mat1b calcReferenceBlock(
    int i,
    int j,
    const cv::Vec2i& aMotionParamsIndices
  ) const;

  cv::Mat1b calcReferenceBlockTrans(
    int i,
    int j,
    const cv::Vec2i& aMotionParamsIndices
  ) const;

  cv::Mat1b calcReferenceBlockDual(
    int i,
    int j,
    const cv::Vec2i& aMotionParamsIndices
  ) const;

 cv::Mat2i calcReferencePoints(
    int i,
    int j,
    double dr,
    double dl
  ) const;

  cv::Vec2d scaleAndRotate(
    int x,
    int y,
    double dr,
    double dl
  ) const;

  uint8_t calcBilinearValue(
    const cv::Vec2d& aPoint
  ) const;

  const cv::Mat1b mBlockTypeMap;
};
