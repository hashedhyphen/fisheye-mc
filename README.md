# fisheye-mc

## Overview
This code provides an implementation of the research paper:

```
加藤 諒，石川知一，亀田裕介，松田一朗，伊東 晋，
「Direct Fisheye形式で記録された全天球動画像の動き補償予測」，
電子情報通信学会総合大会講演論文集，No.D-11-24，(平成29年3月)．
```

## Requirements
* CMake 2.8.11
* g++ 4.8.5
* OpenCV 3.1

## How to build
`cmake -H. -Bbuild`
`cd build ; make`

## How to run
`./fisheye-mc reference.pgm current.pgm`

Sample images are available in the directory `dat`.

## License
3-clause BSD License
([OpenCV License](http://opencv.org/license.html))
