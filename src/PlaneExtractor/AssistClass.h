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
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "ANN.h"

//#include "EightTreePlane.h"


#define minf(a,b)            (((a) < (b)) ? (a) : (b))

using namespace cv;  
using namespace std;


struct PlanePa
{
	int isPlane;  // 0 非平面 1 平面
	cv::Point3d planeNormal;
	cv::Point3d planePoint;
	double lamda;
	int inlinePointSize;       //记录当前的平面是由多少个点拟合的
	double mse;
	float sigema;
	float residual;     
	float maxDistance; 
};
struct TwoDemensionGrid
{
	vector<int> gridPointIndex;
};


struct Grid
{	
	int superGridCluster;
	vector<int> gridPointIndex;
	std::vector<int> neiGridIndex;
};

struct SuperGrid
{
	int planePointSize;
	int superGridCluster;
	vector<Point3i> indexPoint; // x = xcol ; y = yrow; z = zele;
	std::vector<int> planePoints; //包含的所有的点的索引
	vector<int> neiGridIndex;   // 存储和该supergrid 相邻的领域grid
	PlanePa planeParameters;

};

struct MergeEnergy
{
	double costEnergy;
	double distanEnergy;
	double mergeDistance;
	int indexOne;
	int indexTwo;
};

struct GridNormal
{
	int isPlane;
	int superGridIndex;
	double nVectorA;
	double nVectorB;
	double nVectorC;
	double angleZ;     // 该法向量和Z轴的夹角
};

struct polyLine
{
	std::vector<cv::Point3d> polyVetices;
	double a;
	double b;
	double c;
	double d;   // 面方程 ax + by + cz + d = 0;  后面要生成模拟DSM
	double minZ;
	double maxZ;
	int initialIndex;   //记录该polyline在原来的未排序中排第几位
};

class AssistClass
{
public:
	AssistClass(void);
	~AssistClass(void);

	void readGOPointClouds(char* inputRoad, std::vector<Point3d> &lidarPoints, std::vector<std::vector<int>> &clusters);
	void readXYZPointClouds(char* inputRoad, std::vector<std::vector<Point3d>> &lidarPoints);
	void readBuildingXYZPointClouds(char* inputRoad, std::vector<Point3d> &lidarPoints);

	void gravityPoint(std::vector<Point3d> &lidarPoint, double *minXYZ,double *maxXYZ);
	void gravityPointInverse(std::vector<Point3d> &lidarPoint,std::vector<Point3f> &lidarPointGravity,double *minXYZ,double *maxXYZ);

	void rotateAllLidarPoints(std::vector<Point3d> &lidarPoint);
	void twoDemensionGridLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<TwoDemensionGrid> &lidarPointTwoGrid, Mat &pointImage, double factor);
	void findRotateRect(Mat &pointImage, RotatedRect &minRect);
	void computeRotateMatrix(RotatedRect &minRect, Mat &currentRotateMatrix);
	void rotateLidarPoint(std::vector<Point3d> &preBlockLidarPoint,  Mat &rotatedMatrix);

	void minmaxVector(std::vector<Point3d> &lidarPoint, double* minXYZ, double* maxXYZ);
	void gridLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, Point3i &featurePoint, Point3i &leftPoint, float factorGrid, int treeTimes, double *minXYZ,double *maxXYZ);
	void constructNeiGrid(std::vector<Grid> &lidarPointGrid, Point3i &featurePoint);

	void writeLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
		, Point3i &featurePoint,Point3i &leftPoint, double factorGrid, char* outPutRoad, bool isNormal = false, bool isPlaneRect = false);
	void writeInitialPoint(std::vector<Point3d> &lidarPoint,char* outPutRoad);
	void writeSegmentePoint(std::vector<std::vector<Point3d>> &lidarSegmentationPoint, char* outPutRoad);

	void normalVector(std::vector<SuperGrid> &lidarSuperGrid, std::vector<GridNormal> &lidaSuperGridNormal);

	void computeDiffNumber(vector<int> indexClusteTemp, vector<int> &resultIndex);

	double computeNormalAngle(cv::Point3d curNormal, cv::Point3d neiNormal);

	double computInnerProduct(cv::Point3d curNormal, cv::Point3d neiNormal);

	double computeDistaceToPlane(cv::Point3d &planeNoraml, cv::Point3d &planePoint, cv::Point3d &neiPoint);

	void writeRealClusters(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &clusters, char* outPutRoad);
	void writeClusters(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, char* outPutRoad, bool isNormal = false);

	void buildPointPlaneAdjacentGraph(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<SuperGrid> &lidarSuperGrid);

	void planeFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters);

	int findNextPoint(double next_x, double next_y, std::vector<std::pair<cv::Point2d, cv::Point2d>> &allEdgeSegments, std::vector<int> &mask);

	void findMinMax(double *minXY, double *maxXY, std::vector<std::vector<cv::Point3d>> &referencePolyLine, std::vector<std::vector<cv::Point3d>> &detectedPolyLine
		,std::vector<polyLine> &refPolyLies, std::vector<polyLine> &detctPolyLies);

	void obtainPlaneEquation(std::vector<polyLine> &detctPolyLines, std::vector<SuperGrid> &detctedPlanes, std::vector<polyLine> &refPolyLines);

	void generateFeatureImage(std::vector<polyLine> refMyPoly, cv::Mat &featureImage, double minX, double minY, double factorGrid);

	bool isAdd(int curTemp, std::vector<int> tempVector);

	bool isBelong(int temp, std::vector<int> refVector);

	void calculateBoundarPrecision(std::vector<cv::Point3d> &gtBoundaryPoints, std::vector<std::vector<cv::Point3d>> &allPlaneEdgePoints);
	
private: 

};

