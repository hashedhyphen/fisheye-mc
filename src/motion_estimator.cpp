#include "motion_estimator.hpp"

#include <iomanip>
#include <iostream>
#include <map>

namespace {

template <typename Ord>
inline
Ord median(Ord a, Ord b, Ord c)
{
  if (a < b) {
    if (b < c) { return b; } // a <  b <  c
    if (a < c) { return c; } // a <  c <= b
               { return a; } // c <= a <  b
  } else {
    if (a < c) { return a; } // b <= a <  c
    if (b < c) { return c; } // b <  c <= a
               { return b; } // c <= b <= a
  }
}

inline
cv::Vec2i median(const cv::Vec2i& a, const cv::Vec2i& b, const cv::Vec2i& c)
{
  return cv::Vec2i(median(a[0], b[0], c[0]), median(a[1], b[1], c[1]));
}

inline int rshiftRound(int x, int a)
{
  return (x + (1 << (a - 1))) >> a;
}

} // unnamed namespace

// simulates H.264/AVC's spiral search
const std::vector<cv::Vec2i> MotionEstimator::kSpiralOffsets = [] {
  std::vector<cv::Vec2i> offsets;

  offsets.emplace_back(cv::Vec2i(0, 0));

  for (auto m = 1; m <= MotionEstimator::kSearchRange; ++m) {
    for (auto n = -m + 1; n < m; ++n) {
      offsets.emplace_back(cv::Vec2i(n, -m));
      offsets.emplace_back(cv::Vec2i(n,  m));
    }
    for (auto n = -m; n <= m; ++n) {
      offsets.emplace_back(cv::Vec2i(-m, n));
      offsets.emplace_back(cv::Vec2i( m, n));
    }
  }

  return offsets;
}();

const std::array<int, 6> MotionEstimator::k6TapFilter
  = { 1, -5, 20, 20, -5, 1 };

// static
cv::Mat1b MotionEstimator::interpolateReference(
  const cv::Mat1b& aReference
)
{
  const auto height = aReference.rows;
  const auto width  = aReference.cols;

  //        0: integer-pel
  // 2, 8, 10:    half-pel
  //   others: quarter-pel
  // -------------
  // | 0| 1| 2| 3|
  // -------------
  // | 4| 5| 6| 7|
  // -------------
  // | 8| 9|10|11|
  // -------------
  // |12|13|14|15|
  // -------------

  cv::Mat1b interpolated(4 * height - 3, 4 * width - 3);

  // integer-pel [0] (copy)
  aReference.forEach([&] (const std::uint8_t val, const int* pos) {
    const auto x = pos[1], y = pos[0];
    interpolated.at<std::uint8_t>(4 * y, 4 * x) = val;
  });

  // half-pel [8] (apply the 6-tap filter vertically)
  for (auto x = 0; x < width; ++x) {
    int tmp;

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>( 0, 4 * x)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>( 0, 4 * x)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>( 0, 4 * x)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>( 4, 4 * x)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>( 8, 4 * x)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(12, 4 * x);
    interpolated.at<std::uint8_t>(2, 4 * x)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>( 0, 4 * x)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>( 0, 4 * x)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>( 4, 4 * x)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>( 8, 4 * x)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(12, 4 * x)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(16, 4 * x);
    interpolated.at<std::uint8_t>(6, 4 * x)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));

    for (auto y = 2; y < height - 3; ++y) {
      tmp = 0;
      for (auto k = 0; k < 6; ++k) {
        tmp += k6TapFilter[k] * interpolated.at<std::uint8_t>(4 * (y + k - 2),
                                                              4 * x);
      }
      interpolated.at<std::uint8_t>(4 * y + 2, 4 * x)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
    }

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * (height - 5), 4 * x)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * (height - 4), 4 * x)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * (height - 3), 4 * x)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * (height - 2), 4 * x)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * (height - 1), 4 * x)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * (height - 1), 4 * x);
    interpolated.at<std::uint8_t>(4 * (height - 3) + 2, 4 * x)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * (height - 4), 4 * x)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * (height - 3), 4 * x)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * (height - 2), 4 * x)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * (height - 1), 4 * x)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * (height - 1), 4 * x)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * (height - 1), 4 * x);
    interpolated.at<std::uint8_t>(4 * (height - 2) + 2, 4 * x)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
  }

  cv::Mat1i rawValuesPos2(height, width - 1); // raw values at position 2

  // half-pel [2] (apply the 6-tap filter horizontally)
  for (auto y = 0; y < height; ++y) {
    int tmp;

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * y,  0)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * y,  0)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * y,  0)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * y,  4)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * y,  8)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * y, 12);
    interpolated.at<std::uint8_t>(4 * y, 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
    rawValuesPos2.at<int>(y, 0) = tmp;

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * y,  0)
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * y,  0)
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * y,  4)
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * y,  8)
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * y, 12)
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * y, 16);
    interpolated.at<std::uint8_t>(4 * y, 6)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
    rawValuesPos2.at<int>(y, 1) = tmp;

    for (auto x = 2; x < width - 3; ++x) {
      tmp = 0;
      for (auto k = 0; k < 6; ++k) {
        tmp += k6TapFilter[k] * interpolated.at<std::uint8_t>(4 * y,
                                                              4 * (x + k - 2));
      }
      interpolated.at<std::uint8_t>(4 * y, 4 * x + 2)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
      rawValuesPos2.at<int>(y, x) = tmp;
    }

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 5))
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 4))
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 3))
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 2))
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 1))
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 1));
    interpolated.at<std::uint8_t>(4 * y, 4 * (width - 3) + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
    rawValuesPos2.at<int>(y, width - 3) = tmp;

    tmp = k6TapFilter[0] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 4))
        + k6TapFilter[1] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 3))
        + k6TapFilter[2] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 2))
        + k6TapFilter[3] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 1))
        + k6TapFilter[4] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 1))
        + k6TapFilter[5] * interpolated.at<std::uint8_t>(4 * y, 4 * (width - 1));
    interpolated.at<std::uint8_t>(4 * y, 4 * (width - 2) + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 5));
    rawValuesPos2.at<int>(y, width - 2) = tmp;
  }

  // half-pel [10] (apply the 6-tap filter vertically to rawValuesPos2)
  for (auto x = 0; x < width - 1; ++x) {
    int tmp;

    tmp = k6TapFilter[0] * rawValuesPos2.at<int>(0, x)
        + k6TapFilter[1] * rawValuesPos2.at<int>(0, x)
        + k6TapFilter[2] * rawValuesPos2.at<int>(0, x)
        + k6TapFilter[3] * rawValuesPos2.at<int>(1, x)
        + k6TapFilter[4] * rawValuesPos2.at<int>(2, x)
        + k6TapFilter[5] * rawValuesPos2.at<int>(3, x);
    interpolated.at<std::uint8_t>(2, 4 * x + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 10));

    tmp = k6TapFilter[0] * rawValuesPos2.at<int>(0, x)
        + k6TapFilter[1] * rawValuesPos2.at<int>(0, x)
        + k6TapFilter[2] * rawValuesPos2.at<int>(1, x)
        + k6TapFilter[3] * rawValuesPos2.at<int>(2, x)
        + k6TapFilter[4] * rawValuesPos2.at<int>(3, x)
        + k6TapFilter[5] * rawValuesPos2.at<int>(4, x);
    interpolated.at<std::uint8_t>(6, 4 * x + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 10));

    for (auto y = 2; y < height - 3; ++y) {
      tmp = 0;
      for (auto k = 0; k < 6; ++k) {
        tmp += k6TapFilter[k] * rawValuesPos2.at<int>(y + k - 2, x);
      }
      interpolated.at<std::uint8_t>(4 * y + 2, 4 * x + 2)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 10));
    }

    tmp = k6TapFilter[0] * rawValuesPos2.at<int>(height - 5, x)
        + k6TapFilter[1] * rawValuesPos2.at<int>(height - 4, x)
        + k6TapFilter[2] * rawValuesPos2.at<int>(height - 3, x)
        + k6TapFilter[3] * rawValuesPos2.at<int>(height - 2, x)
        + k6TapFilter[4] * rawValuesPos2.at<int>(height - 1, x)
        + k6TapFilter[5] * rawValuesPos2.at<int>(height - 1, x);
    interpolated.at<std::uint8_t>(4 * (height - 3) + 2, 4 * x + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 10));

    tmp = k6TapFilter[0] * rawValuesPos2.at<int>(height - 4, x)
        + k6TapFilter[1] * rawValuesPos2.at<int>(height - 3, x)
        + k6TapFilter[2] * rawValuesPos2.at<int>(height - 2, x)
        + k6TapFilter[3] * rawValuesPos2.at<int>(height - 1, x)
        + k6TapFilter[4] * rawValuesPos2.at<int>(height - 1, x)
        + k6TapFilter[5] * rawValuesPos2.at<int>(height - 1, x);
    interpolated.at<std::uint8_t>(4 * (height - 2) + 2, 4 * x + 2)
      = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 10));
  }

  // quarter-pel [1, 3, 9, 11] (horizontal bilinear)
  for (auto y = 0; y < interpolated.rows; y += 2) {
    for (auto x = 1; x < interpolated.cols; x += 2) {
      const int tmp = interpolated.at<std::uint8_t>(y, x - 1)
                    + interpolated.at<std::uint8_t>(y, x + 1);
      interpolated.at<std::uint8_t>(y, x)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));
    }
  }

  // quarter-pel [4, 6, 12, 14] (vertical bilinear)
  for (auto y = 1; y < interpolated.rows; y += 2) {
    for (auto x = 0; x < interpolated.cols; x += 2) {
      const int tmp = interpolated.at<std::uint8_t>(y - 1, x)
                    + interpolated.at<std::uint8_t>(y + 1, x);
      interpolated.at<std::uint8_t>(y, x)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));
    }
  }

  // quarter-pel [5, 7, 13, 15] (diagonal bilinear)
  for (auto y = 2; y < interpolated.rows; y += 4) {
    for (auto x = 2; x < interpolated.cols; x += 4) {
      int tmp;

      // [5]
      tmp = interpolated.at<std::uint8_t>(y - 2, x)
          + interpolated.at<std::uint8_t>(y, x - 2);
      interpolated.at<std::uint8_t>(y - 1, x - 1)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));

      // [7]
      tmp = interpolated.at<std::uint8_t>(y - 2, x)
          + interpolated.at<std::uint8_t>(y, x + 2);
      interpolated.at<std::uint8_t>(y - 1, x + 1)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));

      // [13]
      tmp = interpolated.at<std::uint8_t>(y, x - 2)
          + interpolated.at<std::uint8_t>(y + 2, x);
      interpolated.at<std::uint8_t>(y + 1, x - 1)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));

      // [15]
      tmp = interpolated.at<std::uint8_t>(y, x + 2)
          + interpolated.at<std::uint8_t>(y + 2, x);
      interpolated.at<std::uint8_t>(y + 1, x + 1)
        = cv::saturate_cast<std::uint8_t>(rshiftRound(tmp, 1));
    }
  }

  return interpolated;
}

// static
cv::Mat1b MotionEstimator::createCircleMask(
  int aFrameSize,
  int aBlockSize
)
{
  const auto radius = aFrameSize / 2;
  const cv::Point center(aFrameSize / 2, aFrameSize / 2);

  cv::Mat1b mask(aFrameSize, aFrameSize);

  for (auto j = 0; j < aFrameSize / aBlockSize; ++j) {
    for (auto i = 0; i < aFrameSize / aBlockSize; ++i) {
      auto block = mask(cv::Rect(aBlockSize * i, aBlockSize * j,
                                 aBlockSize, aBlockSize));
      block = (std::hypot(aBlockSize * (i    ) - center.x,
                          aBlockSize * (j    ) - center.y) < radius
            || std::hypot(aBlockSize * (i + 1) - center.x,
                          aBlockSize * (j    ) - center.y) < radius
            || std::hypot(aBlockSize * (i    ) - center.x,
                          aBlockSize * (j + 1) - center.y) < radius
            || std::hypot(aBlockSize * (i + 1) - center.x,
                          aBlockSize * (j + 1) - center.y) < radius)
            ? 255 : 0;
    }
  }

  return mask;
}

// static
cv::Mat1b MotionEstimator::createIndicesCircleMask(
  int aFrameSize,
  int aBlockSize
)
{
  const auto radius = aFrameSize / 2;
  const cv::Point center(aFrameSize / 2, aFrameSize / 2);

  cv::Mat1b indicesMask(aFrameSize / aBlockSize, aFrameSize / aBlockSize);
  indicesMask.forEach([&] (std::uint8_t& val, const int* pos) {
    const auto i = pos[1], j = pos[0];
    val = (std::hypot(aBlockSize * (i    ) - center.x,
                      aBlockSize * (j    ) - center.y) < radius
        || std::hypot(aBlockSize * (i + 1) - center.x,
                      aBlockSize * (j    ) - center.y) < radius
        || std::hypot(aBlockSize * (i    ) - center.x,
                      aBlockSize * (j + 1) - center.y) < radius
        || std::hypot(aBlockSize * (i + 1) - center.x,
                      aBlockSize * (j + 1) - center.y) < radius)
        ? 255 : 0;
  });

  return indicesMask;
}

MotionEstimator::MotionEstimator(
  const cv::Mat1b& aCurrentFrame,
  const cv::Mat1b& aReferenceFrame,
  int aBlockSize
) : mCurrentFrame(aCurrentFrame)
  , mReferenceFrame(aReferenceFrame)
  , mInterpolatedFrame(MotionEstimator::interpolateReference(aReferenceFrame))
  , mPredictedFrame(aCurrentFrame.rows, aCurrentFrame.cols)
  , mBlockSize(aBlockSize)
  , mCircleMask(MotionEstimator::createCircleMask(
                aCurrentFrame.rows, aBlockSize))
  , mIndicesCircleMask(MotionEstimator::createIndicesCircleMask(
                       aCurrentFrame.rows, aBlockSize))
{
  mMotionParamsIndicesMap = cv::Mat2i(mCurrentFrame.rows / mBlockSize,
                                      mCurrentFrame.cols / mBlockSize);
}

cv::Mat1b MotionEstimator::generatePredictedFrame()
{
  for (auto j = 0; j < mMotionParamsIndicesMap.rows; ++j) {
    for (auto i = 0; i < mMotionParamsIndicesMap.cols; ++i) {
      std::clog << "Estimating... "
                << mMotionParamsIndicesMap.cols * j + i << "/"
                << mMotionParamsIndicesMap.cols * mMotionParamsIndicesMap.rows;

      const auto indices = estimateMotionParamsIndices(i, j);
      mMotionParamsIndicesMap.at<cv::Vec2i>(j, i) = indices;

      calcReferenceBlock(i, j, indices)
      .copyTo(mPredictedFrame(cv::Rect(mBlockSize * i, mBlockSize * j,
                                       mBlockSize, mBlockSize)));

      std::clog << "\r\x1b[K\x1b[m";
    }
  }

  return mPredictedFrame;
}

cv::Vec2i MotionEstimator::estimateMotionParamsIndices(
  int i,
  int j
) const
{
  const auto currentBlock = mCurrentFrame(
    cv::Rect(mBlockSize * i, mBlockSize * j, mBlockSize, mBlockSize));

  auto betterIndices = kSpiralOffsets[0];
  auto minDistortion = std::numeric_limits<Distortion>::max();

  // corresponds to H.264's integer-pel search
  for (const auto& offset: kSpiralOffsets) {
    const auto candidate = 4 * offset;

    const auto referenceBlock = calcReferenceBlock(i, j, candidate);

    const auto distortion = static_cast<Distortion>(
      cv::norm(currentBlock, referenceBlock, cv::NORM_L1)); // SAD

    if (distortion == 0) { return candidate; }

    if (distortion < minDistortion) {
      betterIndices = candidate;
      minDistortion = distortion;
    }
  }

  auto bestIndices = betterIndices;

  // corresponds to H.264's half-pel & quarter-pel search
  const auto searchPoints = (2 * 3 + 1) * (2 * 3 + 1);
  for (auto k = 1; k < searchPoints; ++k) {
    const auto candidate = betterIndices + kSpiralOffsets[k];

    const auto referenceBlock = calcReferenceBlock(i, j, candidate);

    const auto distortion = static_cast<Distortion>(
      cv::norm(currentBlock, referenceBlock, cv::NORM_L1)); // SAD

    if (distortion == 0) { return candidate; }

    if (distortion < minDistortion) {
      bestIndices = candidate;
      minDistortion = distortion;
    }
  }

  return bestIndices;
}

void MotionEstimator::dumpSummary() const
{
  dumpPSNR();
  dumpEntropies();
}

void MotionEstimator::dumpPSNR() const
{
  const auto diff = std::sqrt(
    cv::norm(mCurrentFrame, mPredictedFrame, cv::NORM_L2SQR, mCircleMask)
    / cv::countNonZero(mCircleMask));

  const auto psnr = 20 * std::log10(255. / (diff + DBL_EPSILON));

  std::cout << "PSNR (masked): "
            << std::fixed << std::setprecision(2)
            << psnr << " [dB]" << std::endl;
}

cv::Vec2i MotionEstimator::deriveMedianPredictor(
  int i,
  int j
) const
{
  if (isCodingBlock(i - 1, j)) {
    if (isCodingBlock(i, j - 1)) {
      return isCodingBlock(i + 1, j - 1)
        ? median(mMotionParamsIndicesMap.at<cv::Vec2i>(j    , i - 1),
                 mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i    ),
                 mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i + 1))
        : mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i);
    } else {
      return isCodingBlock(i + 1, j - 1)
        ? mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i + 1)
        : mMotionParamsIndicesMap.at<cv::Vec2i>(j    , i - 1);
    }
  } else {
    if (isCodingBlock(i, j - 1)) {
      return mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i);
    } else {
      return isCodingBlock(i + 1, j - 1)
        ? mMotionParamsIndicesMap.at<cv::Vec2i>(j - 1, i + 1)
        : cv::Vec2i(0, 0);
    }
  }
}

void MotionEstimator::dumpEntropies() const
{
  std::map<int, int> counters[2];
  double entropies[2]; // bits/symbol

  for (auto j = 0; j < mMotionParamsIndicesMap.rows; ++j) {
    for (auto i = 0; i < mMotionParamsIndicesMap.cols; ++i) {
      if (mIndicesCircleMask.at<std::uint8_t>(j, i) == 0) { continue; }

      const auto current = mMotionParamsIndicesMap.at<cv::Vec2i>(j, i);
      const auto predictor = deriveMedianPredictor(i, j);
      ++counters[0][current[0] - predictor[0]];
      ++counters[1][current[1] - predictor[1]];
    }
  }

  for (auto type = 0; type < 2; ++type) {
    for (const auto& pair: counters[type]) {
      const auto p = static_cast<double>(pair.second)
                   / cv::countNonZero(mIndicesCircleMask);
      entropies[type] -= p * std::log2(p);
    }
  }

  std::cout << "Entropy (masked): " << std::scientific << std::setprecision(3)
            << (entropies[0] + entropies[1]) / (mBlockSize * mBlockSize)
            << " [bits/pel]" << std::endl;

  for (auto type = 0; type < 2; ++type) {
    std::cout << "=> Param" << std::to_string(type) << ": "
              << std::scientific << std::setprecision(3)
              << entropies[type] / (mBlockSize * mBlockSize)
              << " [bits/pel]" << std::endl;
  }
}

cv::Mat1b MotionEstimator::visualizePredictionErrors() const
{
  cv::Mat1b diff;
  cv::absdiff(mCurrentFrame, mPredictedFrame, diff);
  return diff;
}

cv::Mat3b MotionEstimator::visualizeMotionIndices() const
{
  cv::Mat3b bgr(mPredictedFrame.rows, mPredictedFrame.cols);

  // copy
  bgr.forEach([&] (cv::Vec3b& pel, const int* pos) {
    const auto x = pos[1], y = pos[0];
    pel[0] = pel[1] = pel[2] = mPredictedFrame.at<std::uint8_t>(y, x);
  });

  // draw a grid
  for (auto j = 1; j < mMotionParamsIndicesMap.rows; ++j) {
    const cv::Point start(0, mBlockSize * j);
    const cv::Point end(bgr.cols - 1, mBlockSize * j);
    cv::line(bgr, start, end, cv::Scalar(255, 50, 50), 1);
  }
  for (auto i = 1; i < mMotionParamsIndicesMap.cols; ++i) {
    const cv::Point start(mBlockSize * i, 0);
    const cv::Point end(mBlockSize * i, bgr.rows - 1);
    cv::line(bgr, start, end, cv::Scalar(255, 50, 50), 1);
  }

  for (auto j = 0; j < mMotionParamsIndicesMap.rows; ++j) {
    for (auto i = 0; i < mMotionParamsIndicesMap.cols; ++i) {
      if (mIndicesCircleMask.at<std::uint8_t>(j, i) == 0) { continue; }

      // draw an arrow
      const cv::Point center(mBlockSize * i + mBlockSize / 2,
                             mBlockSize * j + mBlockSize / 2);
      const auto indices = mMotionParamsIndicesMap.at<cv::Vec2i>(j, i);
      cv::line(bgr, center,
               cv::Point(center.x + indices[0] / 4,
                         center.y + indices[1] / 4),
               cv::Scalar(0, 0, 255), 1);
    }
  }

  return bgr;
}
