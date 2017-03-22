#include "radial.hpp"

#include <cmath>

const double Radial::kSearchStepDr = 0.19;
const double Radial::kSearchStepDl = 0.19;

// static
cv::Mat1b Radial::createBlockTypeMap(
  int aFrameSize,
  int aBlockSize
)
{
  const auto innerRadius = 32;
  const auto outerRadius = aFrameSize / 2;
  const cv::Point center(aFrameSize / 2, aFrameSize / 2);

  cv::Mat1b blockTypeMap(aFrameSize / aBlockSize, aFrameSize / aBlockSize);
  blockTypeMap.forEach([&] (std::uint8_t& val, const int* pos) {
    const auto i = pos[1], j = pos[0];

    if (std::hypot(aBlockSize * (i    ) - center.x,
                   aBlockSize * (j    ) - center.y) < innerRadius
     || std::hypot(aBlockSize * (i + 1) - center.x,
                   aBlockSize * (j    ) - center.y) < innerRadius
     || std::hypot(aBlockSize * (i    ) - center.x,
                   aBlockSize * (j + 1) - center.y) < innerRadius
     || std::hypot(aBlockSize * (i + 1) - center.x,
                   aBlockSize * (j + 1) - center.y) < innerRadius) {
      val = static_cast<std::uint8_t>(BlockType::eTrans);
      return;
    }

    if (std::hypot(aBlockSize * (i    ) - center.x,
                   aBlockSize * (j    ) - center.y) < outerRadius
     || std::hypot(aBlockSize * (i + 1) - center.x,
                   aBlockSize * (j    ) - center.y) < outerRadius
     || std::hypot(aBlockSize * (i    ) - center.x,
                   aBlockSize * (j + 1) - center.y) < outerRadius
     || std::hypot(aBlockSize * (i + 1) - center.x,
                   aBlockSize * (j + 1) - center.y) < outerRadius) {
      val = static_cast<std::uint8_t>(BlockType::eDual);
      return;
    }

    val = static_cast<std::uint8_t>(BlockType::eNone);
  });

  return blockTypeMap;
}

Radial::Radial(
  const cv::Mat1b& aCurrentFrame,
  const cv::Mat1b& aReferenceFrame,
  const int aBlockSize
) : MotionEstimator(aCurrentFrame, aReferenceFrame, aBlockSize)
  , mBlockTypeMap(Radial::createBlockTypeMap( aCurrentFrame.cols, aBlockSize))
{}


cv::Mat1b Radial::calcReferenceBlock(
  int i,
  int j,
  const cv::Vec2i& aMotionParamsIndices
) const
{
  return (mBlockTypeMap.at<std::uint8_t>(j, i)
            == static_cast<std::uint8_t>(BlockType::eTrans))
    ? calcReferenceBlockTrans(i, j, aMotionParamsIndices)
    : calcReferenceBlockDual(i, j, aMotionParamsIndices);
}

cv::Mat1b Radial::calcReferenceBlockTrans(
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

cv::Mat1b Radial::calcReferenceBlockDual(
  int i,
  int j,
  const cv::Vec2i& aMotionParamsIndices
) const
{
  const double dr = kSearchStepDr * aMotionParamsIndices[0];
  const double dl = kSearchStepDl * aMotionParamsIndices[1];

  const auto points = calcReferencePoints(i, j, dr, dl);

  cv::Mat1b block(mBlockSize, mBlockSize);
  block.forEach([&] (std::uint8_t& val, const int* pos) {
    const auto blockX = pos[1], blockY = pos[0];

    const auto point = points.at<cv::Vec2i>(blockY, blockX);
    const auto scaledX = point[0], scaledY = point[1];

    val = isOutOfInterpolatedFrame(scaledX, scaledY)
        ? 0 : mInterpolatedFrame.at<std::uint8_t>(scaledY, scaledX);
  });

  return block;
}

cv::Mat2i Radial::calcReferencePoints(
  int i,
  int j,
  double dr,
  double dl
) const
{
  cv::Mat2i points(mBlockSize, mBlockSize);

  points.forEach([&] (cv::Vec2i& point, const int* pos) {
    const auto x = pos[1], y = pos[0];
    const auto transformed
      = scaleAndRotate(mBlockSize * i + x, mBlockSize * j + y, dr, dl);

    point[0] = static_cast<int>(std::round(4 * transformed[0]));
    point[1] = static_cast<int>(std::round(4 * transformed[1]));
  });

  return points;
}

cv::Vec2d Radial::scaleAndRotate(
  int x,
  int y,
  double dr,
  double dl
) const
{
  const double centerX = mCurrentFrame.cols / 2.;
  const double centerY = mCurrentFrame.rows / 2.;

  const auto polarX = x - centerX;
  const auto polarY = y - centerY;

  const auto r = std::hypot(polarX, polarY);
  const auto t = std::atan2(polarY, polarX);

  const auto dt = dl / r;
  const auto newX = (r + dr) * std::cos(t + dt);
  const auto newY = (r + dr) * std::sin(t + dt);

  return cv::Vec2d(newX + centerX, newY + centerY);
}

std::uint8_t Radial::calcBilinearValue(
  const cv::Vec2d& aPoint
) const
{
  //     x_i x
  // y_i [a]-----[b]
  //      |  |    |
  //    y --[v]----
  //      |  |    |
  //     [c]-----[d]

  const auto x = aPoint[0];
  const auto y = aPoint[1];

  const auto x_i = static_cast<int>(std::floor(x));
  const auto y_i = static_cast<int>(std::floor(y));

  const std::uint8_t a = isOutOfFrame(x_i    , y_i    )
                       ? 0 : mReferenceFrame.at<std::uint8_t>(y_i    , x_i    );
  const std::uint8_t b = isOutOfFrame(x_i + 1, y_i    )
                       ? 0 : mReferenceFrame.at<std::uint8_t>(y_i    , x_i + 1);
  const std::uint8_t c = isOutOfFrame(x_i    , y_i + 1)
                       ? 0 : mReferenceFrame.at<std::uint8_t>(y_i + 1, x_i    );
  const std::uint8_t d = isOutOfFrame(x_i + 1, y_i + 1)
                       ? 0 : mReferenceFrame.at<std::uint8_t>(y_i + 1, x_i + 1);

  const double v = (y_i + 1 - y) * ((x_i + 1 - x) * a + (x - x_i) * b)
                 + (    y - y_i) * ((x_i + 1 - x) * c + (x - x_i) * d);

  return cv::saturate_cast<std::uint8_t>(v);
}
