
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


// #include <pcl/console/parse.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/sample_consensus/ransac.h>
// #include <pcl/sample_consensus/sac_model_plane.h>
// #include <pcl/sample_consensus/sac_model_sphere.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <boost/thread/thread.hpp>


#include "EightTreePlane.h"

#include "AssistClass.h"


#define POINT_NUMBER_THRESHOLD 30//20
#define DISTANCE_THRESHOLD 0.04	//0.05
#define RANSAC_DISTANCE_THRESHOLD 0.01 //0.01
#define RANSAC_POINT_RATIO 0.9   //没用到

#define CURVATURETHRESHOLD 0.1//0.18   越小越少
#define NORMALTHRESHOLD (45*CV_PI)/180

#define  MAX_DISTANCE  1000

#define  MINVALUE  1e-7

using namespace cv;  
using namespace std;

EightTreePlane::EightTreePlane()
{
}

EightTreePlane::~EightTreePlane()
{
}

void EightTreePlane::setData(int inputPointNumberThreshold, double inputMaxDistanceThreshold)
{
	pointNumberThreshold = inputPointNumberThreshold;
	maxDistanceThreshold = inputMaxDistanceThreshold;
}

void EightTreePlane::eightOctreePlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGird
	, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, int treeTimes
	, int tree_PointNumberThreshold, double tree_MaxDistanceThreshold)
{
	setData(tree_PointNumberThreshold, tree_MaxDistanceThreshold);

	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	AssistClass *assit = new AssistClass;

	Point3i leftBottomPoint, rightUpperPoint;
	Point3i eightFeaturePoint;
	for (int i = 0 ; i < featureImageElevation/treeTimes ; i++)
	{
		for (int j = 0 ; j < featureImageHeight/treeTimes ; j++)
		{
			for (int k = 0 ; k < featureImageWidth/treeTimes ; k++)
			{
				leftBottomPoint.x = k*treeTimes; leftBottomPoint.y = j*treeTimes ; leftBottomPoint.z = i*treeTimes;
				rightUpperPoint.x = (k+1)*treeTimes; rightUpperPoint.y = (j+1)*treeTimes; rightUpperPoint.z = (i+1)*treeTimes;
				eightFeaturePoint.x = treeTimes; eightFeaturePoint.y = treeTimes; eightFeaturePoint.z = treeTimes;
				buildEightTree(leftBottomPoint,rightUpperPoint,lidarPoint, lidarPointGrid,lidarSuperGird, featurePoint,eightFeaturePoint, treeTimes);
			}
		}
	}

	for (int i = 0 ; i < lidarSuperGird.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGird[i];
		SuperGrid tempSuperGrid;
		tempSuperGrid.superGridCluster = i;
		int sumPoints = 0;
		std::vector<int> planePointIndex;
		for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
		{
			int gridIndex = currentSuperGrid.indexPoint[j].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[j].y*featureImageWidth + currentSuperGrid.indexPoint[j].x;
			if (lidarPointGrid[gridIndex].gridPointIndex.size())
			{
				sumPoints += lidarPointGrid[gridIndex].gridPointIndex.size();
				tempSuperGrid.indexPoint.push_back(currentSuperGrid.indexPoint[j]);
				for (int k = 0 ; k < lidarPointGrid[gridIndex].gridPointIndex.size() ; k++)
				{
					planePointIndex.push_back(lidarPointGrid[gridIndex].gridPointIndex[k]);
				}
				lidarPointGrid[gridIndex].superGridCluster = i;
			}
		}
		tempSuperGrid.planeParameters = currentSuperGrid.planeParameters;
		tempSuperGrid.planePointSize = sumPoints;
		tempSuperGrid.planePoints = planePointIndex;
		lidarSuperGird[i] = tempSuperGrid;
	}
}


void EightTreePlane::buildEightTree(Point3i &leftBottomPoint, Point3i &rightUpperPoint,std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, 
	std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint, Point3i &eightFeaturePoint, 
	int treeTimes)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	int eightFeatureImageWidth = eightFeaturePoint.x;
	int eightFeatureImageHeight = eightFeaturePoint.y;
	int eightFeatureImageElevation = eightFeaturePoint.z;

	int leftX = leftBottomPoint.x;
	int leftY = leftBottomPoint.y;
	int leftZ = leftBottomPoint.z;

	int rightX = rightUpperPoint.x;
	int rightY = rightUpperPoint.y;
	int rightZ = rightUpperPoint.z;

	int centerX = (int) ((rightX - leftX + 1)/2 + leftX);
	int centerY = (int) ((rightY - leftY + 1)/2 + leftY);
	int centerZ = (int) ((rightZ - leftZ + 1)/2 + leftZ);

	Point3i allLeftBottomPoint[8];
	Point3i allRightUpperPoint[8];   //八叉树分割后，每个区域的左下角的点和右上角的点
	Point3i treeFeaturePoint[8];        //每个区域的长，宽，高

	int indexNumber[8];
	int superIndex[8];
	for (int i = 0 ; i < 8 ; i++)
	{
		indexNumber[i] = 0;
		superIndex[i] = 0;
	}

	std::vector<SuperGrid> curLidarSuperGrid(8);

	for (int i = 0 ; i < eightFeatureImageElevation ; i++)
	{
		for (int j = 0 ; j < eightFeatureImageHeight ; j++)
		{
			for (int k = 0 ; k < eightFeatureImageWidth ; k++)
			{
				int index = (i+leftZ)*(featureImageHeight*featureImageWidth) + (j+leftY)*featureImageWidth + (k+leftX);
				if ( (i+leftZ) < centerZ && (j+leftY) < centerY && (k+leftX) < centerX)// && i < realImageElevation && j < realImageHeight && k < realImageWidth )
				{
					curLidarSuperGrid[0].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//1 最靠近原点的区域
				if ( (i+leftZ) < centerZ && (j+leftY) < centerY && (k+leftX) >= centerX && (k+leftX) < rightX)// && i < realImageElevation && j < realImageHeight)
				{
					curLidarSuperGrid[1].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//2
				if ( (i+leftZ) < centerZ && (j+leftY) >= centerY && (j+leftY) < rightY && (k+leftX) < centerX)// && i < realImageElevation && k < realImageWidth)
				{
					curLidarSuperGrid[2].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//3
				if ( (i+leftZ) < centerZ && (j+leftY) >= centerY && (j+leftY) < rightY && (k+leftX) >= centerX && (k+leftX) < rightX)// && i < realImageElevation) 
				{
					curLidarSuperGrid[3].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//4
				if ( (i+leftZ) >= centerZ && (i+leftZ) < rightZ && (j+leftY) < centerY && (k+leftX) < centerX )//&& j < realImageHeight && k < realImageWidth)
				{
					curLidarSuperGrid[4].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//5
				if ( (i+leftZ) >= centerZ && (i+leftZ) < rightZ && (j+leftY) < centerY && (k+leftX) >= centerX && (k+leftX) < rightX )//&& j < realImageHeight)
				{
					curLidarSuperGrid[5].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//6
				if ( (i+leftZ) >= centerZ && (i+leftZ) < rightZ && (j+leftY) >= centerY && (j+leftY) < rightY && (k+leftX) < centerX )//&& k < realImageWidth)
				{
					curLidarSuperGrid[6].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//7
				if ( (i+leftZ) >= centerZ && (i+leftZ) < rightZ && (j+leftY) >= centerY && (j+leftY) < rightY && (k+leftX) >= centerX && (k+leftX) < rightX)
				{
					curLidarSuperGrid[7].indexPoint.push_back( Point3i((k+leftX),(j+leftY),(i+leftZ)) );/////////////////////////
				}//8 最远离原点的区域
			}
		}
	}


	allLeftBottomPoint[0].x = leftX; allLeftBottomPoint[0].y = leftY; allLeftBottomPoint[0].z = leftZ;
	allRightUpperPoint[0].x = centerX; allRightUpperPoint[0].y = centerY; allRightUpperPoint[0].z = centerZ;

	allLeftBottomPoint[1].x = centerX ; allLeftBottomPoint[1].y = leftY; allLeftBottomPoint[1].z = leftZ;
	allRightUpperPoint[1].x = rightX; allRightUpperPoint[1].y = centerY; allRightUpperPoint[1].z = centerZ;

	allLeftBottomPoint[2].x = leftX; allLeftBottomPoint[2].y = centerY; allLeftBottomPoint[2].z = leftZ;
	allRightUpperPoint[2].x = centerX; allRightUpperPoint[2].y = rightY; allRightUpperPoint[2].z = centerZ;

	allLeftBottomPoint[3].x = centerX; allLeftBottomPoint[3].y = centerY; allLeftBottomPoint[3].z = leftZ;
	allRightUpperPoint[3].x = rightX; allRightUpperPoint[3].y = rightY; allRightUpperPoint[3].z = centerZ;

	allLeftBottomPoint[4].x = leftX; allLeftBottomPoint[4].y = leftY; allLeftBottomPoint[4].z = centerZ;
	allRightUpperPoint[4].x = centerX; allRightUpperPoint[4].y = centerY; allRightUpperPoint[4].z = rightZ;

	allLeftBottomPoint[5].x = centerX; allLeftBottomPoint[5].y = leftY; allLeftBottomPoint[5].z = centerZ;
	allRightUpperPoint[5].x = rightX; allRightUpperPoint[5].y = centerY; allRightUpperPoint[5].z = rightZ;

	allLeftBottomPoint[6].x = leftX; allLeftBottomPoint[6].y = centerY; allLeftBottomPoint[6].z = centerZ;
	allRightUpperPoint[6].x = centerX; allRightUpperPoint[6].y = rightY; allRightUpperPoint[6].z = rightZ;

	allLeftBottomPoint[7].x = centerX; allLeftBottomPoint[7].y = centerY; allLeftBottomPoint[7].z = centerZ;
	allRightUpperPoint[7].x = rightX; allRightUpperPoint[7].y = rightY; allRightUpperPoint[7].z = rightZ;

	if (eightFeaturePoint.x%2 == 0){
		for (int i = 0 ; i < 8 ; i++)
			treeFeaturePoint[i].x = (int) (eightFeaturePoint.x/2);
	}
	if (eightFeaturePoint.x%2 == 1){
		int widthSmall = (int) (eightFeaturePoint.x/2); int widthBig = (int) ((float) eightFeaturePoint.x/2 + 0.8);
		treeFeaturePoint[0].x = widthSmall; treeFeaturePoint[1].x = widthBig; treeFeaturePoint[2].x = widthSmall; treeFeaturePoint[3].x = widthBig;
		treeFeaturePoint[4].x = widthSmall; treeFeaturePoint[5].x = widthBig; treeFeaturePoint[6].x = widthSmall; treeFeaturePoint[7].x = widthBig;
	}
	if (eightFeaturePoint.y%2 == 0){
		for (int i = 0 ; i < 8 ; i++)
			treeFeaturePoint[i].y = (int) (eightFeaturePoint.y/2);
	}
	if (eightFeaturePoint.y%2 == 1){
		int heightSmall = (int) (eightFeaturePoint.y/2); int heightBig = (int) ((float) eightFeaturePoint.y/2 + 0.8);
		treeFeaturePoint[0].y = heightSmall; treeFeaturePoint[1].y = heightSmall; treeFeaturePoint[2].y = heightBig; treeFeaturePoint[3].y = heightBig;
		treeFeaturePoint[4].y = heightSmall; treeFeaturePoint[5].y = heightSmall; treeFeaturePoint[6].y = heightBig; treeFeaturePoint[7].y = heightBig;
	}
	if (eightFeaturePoint.z%2 == 0){
		for (int i = 0 ; i < 8 ; i++)
			treeFeaturePoint[i].z = (int) (eightFeaturePoint.z/2);
	}
	if (eightFeaturePoint.z%2 == 1){
		int eleSmall = (int) (eightFeaturePoint.z/2); int eleBig = (int) ((float) eightFeaturePoint.z/2 + 0.8);
		treeFeaturePoint[0].z = eleSmall; treeFeaturePoint[1].z = eleSmall; treeFeaturePoint[2].z = eleSmall; treeFeaturePoint[3].z = eleSmall;
		treeFeaturePoint[4].z = eleBig; treeFeaturePoint[5].z = eleBig; treeFeaturePoint[6].z = eleBig; treeFeaturePoint[7].z = eleBig;
	}

	for (int i = 0 ; i < 8 ; i++)
	{
		bool checkPlane = false;
		checkPlane = isPlane(lidarPoint,lidarPointGrid, curLidarSuperGrid[i],featurePoint);
		if (!checkPlane && curLidarSuperGrid[i].indexPoint.size() > 1)
		{
			buildEightTree(allLeftBottomPoint[i], allRightUpperPoint[i], lidarPoint, lidarPointGrid, lidarSuperGrid
				, featurePoint,treeFeaturePoint[i], treeTimes);
		}
		if (checkPlane)
		{
			lidarSuperGrid.push_back(curLidarSuperGrid[i]);
		}
	}
}

bool EightTreePlane::isPlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentLidarSuperGrid, Point3i &featurePoint)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	bool result = false;

	bool checkPlane;
	bool checkCurvature;
	bool checkNormal;

	std::vector<Point3d> allSuperGridPoints;
	PlanePa planeParamaterTemp;
	vector<int> pointMaskTemp;

	for (int i = 0 ; i < currentLidarSuperGrid.indexPoint.size() ; i++)
	{
		int gridIndex = currentLidarSuperGrid.indexPoint[i].z*(featureImageHeight*featureImageWidth) + currentLidarSuperGrid.indexPoint[i].y*featureImageWidth + currentLidarSuperGrid.indexPoint[i].x;
		Grid currentLidarGrid = lidarPointGrid[gridIndex];
		for (int j = 0 ; j < currentLidarGrid.gridPointIndex.size() ; j++)
		{
			int pointIndex = currentLidarGrid.gridPointIndex[j];
			allSuperGridPoints.push_back(lidarPoint[pointIndex]);
		}
	}

	if (allSuperGridPoints.size() < pointNumberThreshold)
	{
		allSuperGridPoints.clear();
		return false;
	}
	if (allSuperGridPoints.size() >= pointNumberThreshold)
	{
		checkPlane = robustSvdPlaneFit(allSuperGridPoints,planeParamaterTemp);

		if (checkPlane)
		{
			result = true;
			currentLidarSuperGrid.planeParameters = planeParamaterTemp;
//			currentLidarSuperGrid.inlinePointSize = planeParamaterTemp.inlinePointSize;
		}//if (checkPlane)
		allSuperGridPoints.clear();
		return result;
	}
}

bool EightTreePlane::robustSvdPlaneFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParameters)
{
	int iterateThreshold = 0;
	int iterateNumber = 0;
	int pointSize = (int) allSuperGridPoints.size();

	int realPointSize = pointSize;
	int inlinePointSize = 0;
	pcaPlaneFit(allSuperGridPoints, planeParameters);


	double *planePrecision = new double[3];
	std::vector<double> pointPlaneDistance;
	planeFitAccuracy(allSuperGridPoints, planeParameters, pointPlaneDistance, planePrecision);
	double sigema = planePrecision[0];
	double bigDistance = planePrecision[1];
	double residual = planePrecision[2];
	if (bigDistance <= maxDistanceThreshold)
	//if (residual * residual <= 0.01)
	{
		planeParameters.inlinePointSize = pointSize;
		planeParameters.sigema = planePrecision[0];
		planeParameters.maxDistance = planePrecision[1];
		planeParameters.residual = planePrecision[2];
		planeParameters.mse = planePrecision[0]*planePrecision[0];
		planeParameters.isPlane = 1;
		return true;
	}
	else
	{
		planeParameters.isPlane = 0;
		return false;
	}
	delete planePrecision;
}

void EightTreePlane::pcaPlaneFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters)
{

	int numPoint = allSuperGridPoints.size();
	cv::Matx31d h_mean(0,0,0);
	for (int i = 0 ; i < numPoint; i ++)
	{
		h_mean += cv::Matx31d(allSuperGridPoints[i].x, allSuperGridPoints[i].y, allSuperGridPoints[i].z);
	}
	h_mean *= (1.0/ numPoint);

	cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	for( int i = 0; i < numPoint; i ++ )
	{
		cv::Matx31d hi = cv::Matx31d(allSuperGridPoints[i].x, allSuperGridPoints[i].y, allSuperGridPoints[i].z);
		h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
	}
	h_cov *=( 1.0 / numPoint );

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

	double t = h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] + ( rand()%10 + 1 ) * MINVALUE;
	planeParemeters.lamda = h_cov_evals.row(2).val[0] / t;

	cv::Matx31d normalTemp = h_cov_evectors.row(2).t();
	planeParemeters.planeNormal = cv::Point3d(normalTemp.val[0], normalTemp.val[1], normalTemp.val[2]);
	planeParemeters.planePoint = cv::Point3d(h_mean.val[0], h_mean.val[1], h_mean.val[2]);

}

void EightTreePlane::planeFitAccuracy(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters, std::vector<double> &pointPlaneDistance, double *planePrecision)
{
	int pointSize = (int) allSuperGridPoints.size();
	double planeDistanceTemp;
	for (int i = 0 ; i < allSuperGridPoints.size() ; i++)
	{
		planeDistanceTemp = computePointToPlaneDistance(allSuperGridPoints[i],planeParemeters);
		pointPlaneDistance.push_back(planeDistanceTemp);
	}

	double distanceSum = 0;
	double maxDisntance = 0;
	for (int i = 0 ; i < allSuperGridPoints.size() ; i++)
	{
		distanceSum += pointPlaneDistance[i];
		if (pointPlaneDistance[i] >= maxDisntance)
		{
			maxDisntance = pointPlaneDistance[i];
		}
	}
	double meanDistance = (double) distanceSum/pointSize;
	double sumTemp = 0;
	double sumResidual = 0;
	for (int i = 0 ; i < pointSize ; i++)
	{
		sumTemp += (pointPlaneDistance[i] -meanDistance)*(pointPlaneDistance[i] - meanDistance);
		sumResidual += (pointPlaneDistance[i] * pointPlaneDistance[i]);
	}
	//double mse = (double) sumTemp/pointSize;
	double sigema;
	sigema = sqrtf(sumTemp/pointSize);
	double residual = sqrtf(sumResidual / pointSize);

	planePrecision[0] = sigema;
	planePrecision[1] = maxDisntance;
	planePrecision[2] = residual;

}

bool EightTreePlane::pclRansacAlgorithm( std::vector<Point3d> &gridPoint, PlanePa &planeParameters, double distanceThreshold)
{
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
// 	cloud->resize(gridPoint.size());
// 	for (int i = 0 ; i < gridPoint.size() ; i++)
// 	{
// 		cloud->points[i].x = gridPoint[i].x;
// 		cloud->points[i].y = gridPoint[i].y;
// 		cloud->points[i].z = gridPoint[i].z;
// 	}
// 	std::vector<int> inliers;
// 	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr modelPlane (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
// 	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (modelPlane);
// 	ransac.setDistanceThreshold (distanceThreshold);
// 	ransac.computeModel();
// 	ransac.getInliers(inliers);
// 	
// 	if (!inliers.size())
// 	{
// 		planeParameters.isPlane = 0;
// 		return false;
// 	}
// 
// 	Eigen::VectorXf model_coefficients;
// 	ransac.getModelCoefficients(model_coefficients);
// 
// 	cv::Point3d sumPoints; sumPoints.x = 0 ; sumPoints.y = 0 ; sumPoints.z = 0;
// 	for (int i = 0 ; i < inliers.size() ; i++)
// 	{
// 		sumPoints.x += gridPoint[inliers[i]].x;
// 		sumPoints.y += gridPoint[inliers[i]].y;
// 		sumPoints.z += gridPoint[inliers[i]].z;
// 	}
// 	sumPoints.x *= (1.0/inliers.size());
// 	sumPoints.y *= (1.0/inliers.size());
// 	sumPoints.z *= (1.0/inliers.size());
// 
// 	planeParameters.isPlane = 1;
// 	cv::Point3d normalTemp = cv::Point3d(model_coefficients[0], model_coefficients[1], model_coefficients[2]);
// 	double mu = sqrtf(normalTemp.x*normalTemp.x + normalTemp.y*normalTemp.y + normalTemp.z*normalTemp.z);
// 	normalTemp.x /= mu; normalTemp.y /= mu; normalTemp.z /= mu;
// 	planeParameters.planeNormal = normalTemp;
// 
// 	planeParameters.planePoint = sumPoints;
// 
// 	planeParameters.inlinePointSize = inliers.size();
// 
// 
// 	inliers.clear();
// 	cloud->clear();

	return true;
}


void EightTreePlane::buildAdjacencyRelation(std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint)
{
	std::vector<int> dx(26);
	std::vector<int> dy(26);
	std::vector<int> dz(26);


	AssistClass *assist = new AssistClass;
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		if (i == 26 || i== 53)
		{
			int tiaoshi = 1;
		}
		vector<int> neiGridIndexTemp;
		SuperGrid* currentLidarSuperGrid = &lidarSuperGrid[i];
		int currentSuperGridCluster = currentLidarSuperGrid->superGridCluster;
		int pointGridSize = currentLidarSuperGrid->indexPoint.size();
		if (pointGridSize)
		{
			for (int j = 0 ; j < pointGridSize ; j++)
			{
				int colx = currentLidarSuperGrid->indexPoint[j].x;
				int rowy = currentLidarSuperGrid->indexPoint[j].y;
				int elez = currentLidarSuperGrid->indexPoint[j].z;
				int indexGrid = elez*(featureImageWidth*featureImageHeight) + rowy*featureImageWidth + colx;
				/**************************求该grid 的26领域*********************/
				if (lidarPointGrid[indexGrid].gridPointIndex.size())
				{
					int eightNeiIndex[26];
					int eightRow[26];
					int eightCol[26];
					int eightEle[26];
					for (int k = 0 ; k < 26 ; k++)
					{
						if (k <= 8 )
							eightEle[k] = elez + 1;
						if (k <= 16 && k >= 9)
							eightEle[k] = elez;
						if (k <= 25 && k >= 17)
							eightEle[k] = elez - 1;

						if (k == 0 || k == 1 || k == 2 || k == 9 || k == 10 || k == 11 || k == 17 || k == 18 || k == 19)
							eightRow[k] = rowy - 1;
						if (k == 3 || k == 4 || k == 5 || k == 12 || k == 13 || k == 20 || k == 21 || k == 22)
							eightRow[k] = rowy;
						if (k == 6 || k == 7 || k == 8 || k == 14 || k == 15 || k == 16 || k == 23 || k == 24 || k== 25)
							eightRow[k] = rowy + 1;

						if (k == 0 || k == 3 || k == 6 || k == 9 || k == 12 || k == 14 || k == 17 || k == 20 || k== 23)
							eightCol[k] = colx - 1;
						if (k == 1 || k == 4 || k == 7 || k == 10 || k == 15 || k == 18 || k == 21 || k == 24)
							eightCol[k] = colx;
						if (k == 2 || k == 5 || k == 8 || k == 11 || k == 13 || k == 16 || k == 19 || k == 22 || k== 25)
							eightCol[k] = colx + 1;
					}

					for (int k = 0 ; k < 26 ; k++)
					{
						eightNeiIndex[k] = eightEle[k]*(featureImageWidth*featureImageHeight) + eightRow[k]*featureImageWidth + eightCol[k];
					}
					/**************************求该grid 的26领域*********************/

					for (int k = 0 ; k < 26 ; k++)
					{
						if (eightEle[k] >=0 && eightEle[k] < featureImageElevation && eightRow[k] >= 0 && eightRow[k] < featureImageHeight && eightCol[k] >= 0 && eightCol[k] < featureImageWidth)
						{
							if (lidarPointGrid[eightNeiIndex[k]].superGridCluster != currentSuperGridCluster && lidarPointGrid[eightNeiIndex[k]].gridPointIndex.size())
							{
								if (lidarPointGrid[eightNeiIndex[k]].superGridCluster != -1)
								{
									neiGridIndexTemp.push_back(lidarPointGrid[eightNeiIndex[k]].superGridCluster);
								}
							}
						}
					}
				}// if 
			}//for
			lidarSuperGrid[i].neiGridIndex.clear();
			assist->computeDiffNumber(neiGridIndexTemp,lidarSuperGrid[i].neiGridIndex);
			neiGridIndexTemp.clear();
		}//if
	}//for
}

double EightTreePlane::computePointToPlaneDistance(Point3d pointTemp, PlanePa planeParameter)
{
	double planeDistance = 0;

	cv::Point3d seedToCurPoint;
	seedToCurPoint.x = pointTemp.x - planeParameter.planePoint.x;
	seedToCurPoint.y = pointTemp.y - planeParameter.planePoint.y;
	seedToCurPoint.z = pointTemp.z - planeParameter.planePoint.z;
	double pointDistance = fabs( seedToCurPoint.x * planeParameter.planeNormal.x + seedToCurPoint.y * planeParameter.planeNormal.y + seedToCurPoint.z * planeParameter.planeNormal.z );
	return pointDistance;
}

