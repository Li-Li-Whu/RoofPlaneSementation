#pragma once
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "stdio.h"
#include "vector"
#include "iostream"
#include "fstream"
#include "sstream"
#include "math.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <queue>


using namespace cv;  
using namespace std;

#include "AssistClass.h"
#include "EightTreePlane.h"
#include "PlaneMerge.h"
#include "AddPointSeedGrow.h"




#include "ANN.h"

class FacadeFootprintExtraction
{
public:
	FacadeFootprintExtraction(void);
	~FacadeFootprintExtraction(void);

	void buildingFootPrintExtraction(std::vector<cv::Point3d> &lidarPoints, std::vector<SuperGrid> &detctedPlanes, std::vector<std::vector<cv::Point3d>> &allPlaneEdgePoints, int blockSize, double treeDistanceThreshold, double mergeMSEThreshold, double add_DistanceThreshold, int pointNumThreshold, double lambda);

	void boundaryRelabellingWithInitialCluster(std::vector<cv::Point3d> &lidarPoints, std::vector<std::vector<int>> &clusters, int blockSize, double lambda);

	void kdTreePoints(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance, int kSize, double kDis);

    void relabelingPlanes(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex,double lamda);

	void findAllBoundaryPoint(std::vector<Point3d> &lidarPoint, std::vector<int> pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGrid, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber);

	void filterSmallPlanes(std::vector<SuperGrid> &lidarSuperGrid, int pointNumThreshold);

	void filterVerticalPlanes(std::vector<SuperGrid> &lidarSuperGrid, double angleThreshold);

	double statisticsBoundaryHistogram(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex);

	double statisticsBoundaryHistogramWeight(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex);

	double statisticsBoundaryHistogramAfterChange(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex, int curCluseterIndex, int neiClusterIndex);

	double statisticsBoundaryHistogramAfterChangeWeight(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex, int curCluseterIndex, int neiClusterIndex);




	void newRelablingPlanes(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance);

	void initialEnergy(std::vector<Point3d> &lidarPoint, std::vector<bool> &in_q, std::vector<int> &labels, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allNeiLabels, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance, std::vector<double> &initialBoundaryEnergy);

	void calculateMoveEnergy(std::vector<Point3d> &lidarPoint, std::vector<int> &labels, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance, std::vector<double> &initialBoundaryEnergy, std::vector<std::vector<int>> &allNeiLabels, int iIdx);

	

};

