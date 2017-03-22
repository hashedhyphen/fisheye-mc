#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "h264.hpp"
#include "radial.hpp"

namespace {

void evaluate(
  MotionEstimator& estimator,
  const std::string& type
)
{
  std::cout << type << " (" << estimator.blockSize() << ")" << std::endl;

  const auto start = std::chrono::system_clock::now();
  const auto predicted = estimator.generatePredictedFrame();
  const auto end = std::chrono::system_clock::now();
  std::cout << "Elapsed: " << std::chrono::duration<double>(end - start).count()
            << " [s]" << std::endl;

  estimator.dumpSummary();

  const std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 9 };

  cv::imwrite("dat/predicted_" + type + "_"
              + std::to_string(estimator.blockSize()) + ".png",
              predicted, params);

  cv::imwrite("dat/diff_" + type + "_"
              + std::to_string(estimator.blockSize()) + ".png",
              estimator.visualizePredictionErrors(), params);

  cv::imwrite("dat/motion_" + type + "_"
              + std::to_string(estimator.blockSize()) + ".png",
              estimator.visualizeMotionIndices(), params);

  std::cout << std::endl;
}

} // unnamed namespace

int main(int argc, char** argv)
{
  cv::setNumThreads(4);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " reference.pgm current.pgm" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto reference = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  const auto current   = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

  for (auto size = 4; size <= 32; size *= 2) {
    H264 trans(current, reference, size);
    evaluate(trans, "trans");

    Radial radial(current, reference, size);
    evaluate(radial, "dual");
  }

  return EXIT_SUCCESS;
}
