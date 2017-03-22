#include "h264.hpp"

H264::H264(
  const cv::Mat1b& aCurrentFrame,
  const cv::Mat1b& aReferenceFrame,
  int aBlockSize
) : MotionEstimator(aCurrentFrame, aReferenceFrame, aBlockSize)
{}

cv::Mat1b H264::calcReferenceBlock(
  int i,
  int j,
  const cv::Vec2i& aMotionParamsIndices
) const
{
  const auto originX = 4 * mBlockSize * i + aMotionParamsIndices[0];
  const auto originY = 4 * mBlockSize * j + aMotionParamsIndices[1];

  cv::Mat1b block(mBlockSize, mBlockSize);

  block.forEach([&] (std::uint8_t& val, const int* pos) {
    const auto x = pos[1], y = pos[0];

    const auto referenceX = originX + 4 * x;
    const auto referenceY = originY + 4 * y;

    val = (referenceX < 0 || mInterpolatedFrame.cols <= referenceX ||
           referenceY < 0 || mInterpolatedFrame.rows <= referenceY)
        ? 0 : mInterpolatedFrame.at<std::uint8_t>(referenceY, referenceX);
  });

  return block;
}
