#pragma once

#include "motion_estimator.hpp"

#include <cstdint>
#include <array>

class H264 : public MotionEstimator
{
public:
  H264(
    const cv::Mat1b& aCurrentFrame,
    const cv::Mat1b& aReferenceFrame,
    int aBlockSize
  );

private:
  cv::Mat1b calcReferenceBlock(
    int i,
    int j,
    const cv::Vec2i& aMotionParamsIndices
  ) const;
};
