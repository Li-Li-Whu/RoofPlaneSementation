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
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>


#include <set>

#define  InfiniteMinValue 0.0001

#define  MINVALUE  1e-7


#include "AssistClass.h"


#include "lasreader.h"

#include "ANN.h"


//#include <CGAL\Delaunay_triangulation_2.h>

#include <list>




#ifndef mmax
#define mmax(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef mmin
#define mmin(a,b)            (((a) < (b)) ? (a) : (b))
#endif


using namespace cv;  
using namespace std;

AssistClass::AssistClass(void)
{
}

AssistClass::~AssistClass(void)
{
}

void AssistClass::readGOPointClouds(char* inputRoad, std::vector<Point3d> &lidarPoints, std::vector<std::vector<int>> &clusters)
{
	double x, y, z;
	int l;
	double t1, t2, t3, t4;
	char buffer[100];

	FILE *fpL1 = fopen(inputRoad, "r");

	int maxIndex = 0;
	std::vector<int> clusterIndex;
	while (!feof(fpL1))
	{
		fscanf(fpL1, "%lf %lf %lf %d", &x, &y, &z, &l);
		lidarPoints.push_back(Point3d(x, y, z));
		clusterIndex.push_back(l);
		if (l > maxIndex)
		{
			maxIndex = l;
		}
		FILE *fpTemp = fpL1;
		while (fgets(buffer, 100, fpTemp) == "")
		{
			fpL1 = fpTemp;
		}
	}
	fclose(fpL1);

	clusters.resize(maxIndex + 1);
	for (int i = 0; i < clusterIndex.size(); i++)
	{
		clusters[clusterIndex[i]].push_back(i);
	}

}

void AssistClass::readXYZPointClouds(char* inputRoad, std::vector<std::vector<Point3d>> &lidarPoints)
{
	double x,y,z;
	int l;
	int r,g,b;
	double t1,t2,t3,t4;
	char buffer[100];

	FILE *fpL1 = fopen( inputRoad, "r");

	while(!feof(fpL1))
	{
		fscanf(fpL1,"%lf %lf %lf %d %d %d %d",&x,&y,&z,&r, &g, &b, &l);
		//cout<<l<<endl;
		lidarPoints[l].push_back(Point3d(x,y,z));
		
		FILE *fpTemp=fpL1;
		while (fgets(buffer,100,fpTemp) == "")
		{
			fpL1=fpTemp;
		}
	}
	fclose(fpL1);
}

void AssistClass::readBuildingXYZPointClouds(char* inputRoad, std::vector<Point3d> &lidarPoints)
{
	double x,y,z;
	int r,g,b;
	double t1,t2,t3,t4;
	char buffer[100];

	FILE *fpL1 = fopen( inputRoad, "r");

	while(!feof(fpL1))
	{
		fscanf(fpL1,"%lf %lf %lf %d %d %d",&x,&y,&z,&r, &g, &b);
		lidarPoints.push_back(Point3d(x,y,z));
		FILE *fpTemp=fpL1;
		while (fgets(buffer,100,fpTemp) == "")
		{
			fpL1=fpTemp;
		}
	}
	fclose(fpL1);
}

void AssistClass::writeLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
	, Point3i &featurePoint,Point3i &leftPoint, double factorGrid, char* outPutRoad, bool isNormal,bool isPlaneRect)
{
	int leftX = leftPoint.x;
	int leftY = leftPoint.y;
	int leftZ = leftPoint.z;
	std::vector<GridNormal> lidaSuperGridNormal(lidarSuperGrid.size());

	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	if (isNormal)
	{
		normalVector(lidarSuperGrid,lidaSuperGridNormal);
	}

	ofstream lidarResult(outPutRoad);

	srand((unsigned int)time(0));
	int itemp = 0;
	for (int i = 0 ; i < (int) lidarSuperGrid.size() ; i++)
	{
		int r = rand()%255;
		int g = rand()%255;
		int b = rand()%255;
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (currentSuperGrid.planeParameters.isPlane)
		{
			for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
			{
				int colx = currentSuperGrid.indexPoint[j].x;
				int rowy = currentSuperGrid.indexPoint[j].y;
				int elez = currentSuperGrid.indexPoint[j].z;
				int indexGrid = elez*(featureImageWidth*featureImageHeight) + rowy*featureImageWidth + colx;
				for (int k = 0 ; k < lidarPointGrid[indexGrid].gridPointIndex.size() ; k++)
				{
					int indexPoint = lidarPointGrid[indexGrid].gridPointIndex[k];
					lidarResult<<setiosflags(ios::fixed)<<lidarPoint[indexPoint].x<<" "<<lidarPoint[indexPoint].y<<" "<<lidarPoint[indexPoint].z<<" "<<r<<" "<<g<<" "<<b<<endl;
				}
			}
		}
		if (!currentSuperGrid.planeParameters.isPlane)
		{
			for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
			{
				int colx = currentSuperGrid.indexPoint[j].x;
				int rowy = currentSuperGrid.indexPoint[j].y;
				int elez = currentSuperGrid.indexPoint[j].z;
				int indexGrid = elez*(featureImageWidth*featureImageHeight) + rowy*featureImageWidth + colx;
				for (int k = 0 ; k < lidarPointGrid[indexGrid].gridPointIndex.size() ; k++)
				{
					int indexPoint = lidarPointGrid[indexGrid].gridPointIndex[k];
					lidarResult<<setiosflags(ios::fixed)<<lidarPoint[indexPoint].x<<" "<<lidarPoint[indexPoint].y<<" "<<lidarPoint[indexPoint].z<<" "<<0<<" "<<0<<" "<<255<<endl;
				}
			}
		}


		if (currentSuperGrid.planeParameters.isPlane && currentSuperGrid.indexPoint.size() && isNormal)
		{
			int colx = currentSuperGrid.indexPoint[0].x;
			int rowy = currentSuperGrid.indexPoint[0].y;
			int elez = currentSuperGrid.indexPoint[0].z;
			float centerX = leftX + colx*factorGrid + factorGrid/2;
			float centerY = leftY + rowy*factorGrid + factorGrid/2;
			float centerZ = leftZ + elez*factorGrid + factorGrid/2;

			float lastX = centerX + lidaSuperGridNormal[i].nVectorA;
			float lastY = centerY + lidaSuperGridNormal[i].nVectorB;
			float lastZ = centerZ + lidaSuperGridNormal[i].nVectorC;

			float distanceX = lastX - centerX;
			float distanceY = lastY - centerY;
			float distanceZ = lastZ - centerZ;

			int stepDistance = 100;
			float factorNormalX = distanceX/stepDistance; float factorNormalY = distanceY/stepDistance; float factorNormalZ = distanceZ/stepDistance;

			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = centerX + m*factorNormalX;
				float coordY = centerY + m*factorNormalY;
				float coordZ = centerZ + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
		}
/*		if (currentSuperGrid.planeParameters.isPlane && currentSuperGrid.indexPoint.size() && isPlaneRect)
		{
			PlaneRect currentPlaneRect = lidarPlaneRect[itemp];
			for (int j = 0 ; j < 4 ; j++)
			{
				lidarResult<<setiosflags(ios::fixed)<<currentPlaneRect.fourPeakPoint[j].x<<" "<<currentPlaneRect.fourPeakPoint[j].y<<" "<<currentPlaneRect.fourPeakPoint[j].z<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
			int stepDistance = 100;
			float factorNormalX, factorNormalY, factorNormalZ;
			//0
			factorNormalX = (currentPlaneRect.fourPeakPoint[1].x-currentPlaneRect.fourPeakPoint[0].x)/stepDistance; 
			factorNormalY = (currentPlaneRect.fourPeakPoint[1].y-currentPlaneRect.fourPeakPoint[0].y)/stepDistance; 
			factorNormalZ = (currentPlaneRect.fourPeakPoint[1].z-currentPlaneRect.fourPeakPoint[0].z)/stepDistance;
			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = currentPlaneRect.fourPeakPoint[0].x + m*factorNormalX;
				float coordY = currentPlaneRect.fourPeakPoint[0].y + m*factorNormalY;
				float coordZ = currentPlaneRect.fourPeakPoint[0].z + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
			//1
			factorNormalX = (currentPlaneRect.fourPeakPoint[2].x-currentPlaneRect.fourPeakPoint[0].x)/stepDistance; 
			factorNormalY = (currentPlaneRect.fourPeakPoint[2].y-currentPlaneRect.fourPeakPoint[0].y)/stepDistance; 
			factorNormalZ = (currentPlaneRect.fourPeakPoint[2].z-currentPlaneRect.fourPeakPoint[0].z)/stepDistance;
			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = currentPlaneRect.fourPeakPoint[0].x + m*factorNormalX;
				float coordY = currentPlaneRect.fourPeakPoint[0].y + m*factorNormalY;
				float coordZ = currentPlaneRect.fourPeakPoint[0].z + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
			//2
			factorNormalX = (currentPlaneRect.fourPeakPoint[2].x-currentPlaneRect.fourPeakPoint[3].x)/stepDistance; 
			factorNormalY = (currentPlaneRect.fourPeakPoint[2].y-currentPlaneRect.fourPeakPoint[3].y)/stepDistance; 
			factorNormalZ = (currentPlaneRect.fourPeakPoint[2].z-currentPlaneRect.fourPeakPoint[3].z)/stepDistance;
			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = currentPlaneRect.fourPeakPoint[3].x + m*factorNormalX;
				float coordY = currentPlaneRect.fourPeakPoint[3].y + m*factorNormalY;
				float coordZ = currentPlaneRect.fourPeakPoint[3].z + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
			//3
			factorNormalX = (currentPlaneRect.fourPeakPoint[1].x-currentPlaneRect.fourPeakPoint[3].x)/stepDistance; 
			factorNormalY = (currentPlaneRect.fourPeakPoint[1].y-currentPlaneRect.fourPeakPoint[3].y)/stepDistance; 
			factorNormalZ = (currentPlaneRect.fourPeakPoint[1].z-currentPlaneRect.fourPeakPoint[3].z)/stepDistance;
			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = currentPlaneRect.fourPeakPoint[3].x + m*factorNormalX;
				float coordY = currentPlaneRect.fourPeakPoint[3].y + m*factorNormalY;
				float coordZ = currentPlaneRect.fourPeakPoint[3].z + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
			itemp++;
		}*/
	}
	lidaSuperGridNormal.clear();
}


void AssistClass::writeRealClusters(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &clusters, char* outPutRoad)
{
	srand((unsigned int)time(0));
	int itemp = 0;
	ofstream lidarResult(outPutRoad);
	for (int i = 1; i < clusters.size(); i++)
	{
		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;

		for (int j = 0; j < clusters[i].size(); j++)
		{
			lidarResult << setiosflags(ios::fixed) << lidarPoint[clusters[i][j]].x << " " << lidarPoint[clusters[i][j]].y << " " << lidarPoint[clusters[i][j]].z << " " << r << " " << g << " " << b << endl;
		}
	}
}


void AssistClass::writeClusters(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, char* outPutRoad, bool isNormal)
{
	std::vector<std::vector<int>> clusters(lidarSuperGrid.size());
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	//for (int i = 0; i < 14; i++)
	{
		clusters[i] = lidarSuperGrid[i].planePoints;
	}

	std::vector<GridNormal> lidaSuperGridNormal(lidarSuperGrid.size());
	if (isNormal)
	{
		normalVector(lidarSuperGrid,lidaSuperGridNormal);
	}

// 	ofstream lidarResult1("D:\\125.txt");
// 	int r = rand()%255;
// 	int g = rand()%255;
// 	int b = rand()%255;
// 	int iTemp = 125;
// 	for (int j = 0 ; j < clusters[iTemp].size() ; j++)
// 	{
// 		lidarResult1<<setiosflags(ios::fixed)<<lidarPoint[clusters[iTemp][j]].x<<" "<<lidarPoint[clusters[iTemp][j]].y<<" "<<lidarPoint[clusters[iTemp][j]].z<<" "<<r<<" "<<g<<" "<<b<<endl;
// 	}


	srand((unsigned int)time(0));
	int itemp = 0;
	ofstream lidarResult(outPutRoad);
	for (int i = 0 ; i <  clusters.size() ; i++)
	{
		int r = rand()%255;
		int g = rand()%255;
		int b = rand()%255;

		for (int j = 0 ; j < clusters[i].size() ; j++)
		{
			lidarResult<<setiosflags(ios::fixed)<<lidarPoint[clusters[i][j]].x<<" "<<lidarPoint[clusters[i][j]].y<<" "<<lidarPoint[clusters[i][j]].z<<" "<<r<<" "<<g<<" "<<b<<endl;
		}

		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (currentSuperGrid.planeParameters.isPlane && currentSuperGrid.indexPoint.size() && isNormal)
		{

			double centerX = currentSuperGrid.planeParameters.planePoint.x;
			double centerY = currentSuperGrid.planeParameters.planePoint.y;
			double centerZ = currentSuperGrid.planeParameters.planePoint.z;

			float lastX = centerX + lidaSuperGridNormal[i].nVectorA;
			float lastY = centerY + lidaSuperGridNormal[i].nVectorB;
			float lastZ = centerZ + lidaSuperGridNormal[i].nVectorC;

			float distanceX = lastX - centerX;
			float distanceY = lastY - centerY;
			float distanceZ = lastZ - centerZ;

			int stepDistance = 100;
			float factorNormalX = distanceX/stepDistance; float factorNormalY = distanceY/stepDistance; float factorNormalZ = distanceZ/stepDistance;

			for (int m = 0 ; m < stepDistance ; m++)
			{
				float coordX = centerX + m*factorNormalX;
				float coordY = centerY + m*factorNormalY;
				float coordZ = centerZ + m*factorNormalZ;
				lidarResult<<setiosflags(ios::fixed)<<coordX<<" "<<coordY<<" "<<coordZ<<" "<<r<<" "<<g<<" "<<b<<endl;
			}
		}

	}


}

void AssistClass::writeInitialPoint(std::vector<Point3d> &lidarPoint,char* outPutRoad)
{
	ofstream lidarResult(outPutRoad);
	int r = 255 ; int g = 0 ; int b = 0;

	for (int i = 0 ; i < (int) lidarPoint.size() ; i++)
	{
		lidarResult<<setiosflags(ios::fixed)<<lidarPoint[i].x<<" "<<lidarPoint[i].y<<" "<<lidarPoint[i].z<<" "<<r<<" "<<g<<" "<<b<<endl;
	}
}

void AssistClass::writeSegmentePoint(std::vector<std::vector<Point3d>> &lidarSegmentationPoint, char* outPutRoad)
{
	ofstream lidarResult(outPutRoad);
	srand((unsigned int)time(0));
	for (int i = 0 ; i < lidarSegmentationPoint.size() ; i++)
	{
// 		int r = rand()%255;
// 		int g = rand()%255;
// 		int b = rand()%255;
		int r = 255;
		int g = 0;
		int b = 0;
		std::vector<Point3d> currentSegmentation;
		currentSegmentation = lidarSegmentationPoint[i];
		for (int j = 0 ; j < currentSegmentation.size() ; j++)
		{
			lidarResult<<setiosflags(ios::fixed)<<currentSegmentation[j].x<<" "<<currentSegmentation[j].y<<" "<<currentSegmentation[j].z<<" "<<r<<" "<<g<<" "<<b<<endl;
		}
	}
}

void AssistClass::minmaxVector(std::vector<Point3d> &lidarPoint, double* minXYZ, double* maxXYZ)
{
	int lidarPointSize=lidarPoint.size();
	double minX=10000000000, minY=10000000000,minZ=10000000000;
	double maxX=-1000000, maxY=-1000000,maxZ=-1000000;
	for (int i = 0 ; i < lidarPointSize; ++i)
	{
		double lidarPointX=lidarPoint[i].x;
		double lidarPointY=lidarPoint[i].y;
		double lidarPointZ=lidarPoint[i].z;
		if (lidarPointX<minX) minX=lidarPointX;
		if (lidarPointX>maxX) maxX=lidarPointX;

		if (lidarPointY<minY) minY=lidarPointY;
		if (lidarPointY>maxY) maxY=lidarPointY;

		if (lidarPointZ<minZ) minZ=lidarPointZ;
		if (lidarPointZ>maxZ) maxZ=lidarPointZ;
	}
	minXYZ[0] =minX;
	minXYZ[1] =minY;
	minXYZ[2] =minZ;

	maxXYZ[0] =maxX;
	maxXYZ[1] =maxY;
	maxXYZ[2] =maxZ;
}

void AssistClass::gravityPoint(std::vector<Point3d> &lidarPoint, double *minXYZ,double *maxXYZ)
{
	minmaxVector(lidarPoint, minXYZ, maxXYZ);

	double minX = minXYZ[0];
	double minY = minXYZ[1];
	double minZ = minXYZ[2];

	int numPoint = lidarPoint.size();
	std::vector<Point3d> lidarPointGravity(numPoint);
	Point3f pointTemp;

	for (int i = 0 ; i < numPoint ; i++)
	{
		pointTemp.x = lidarPoint[i].x - minX;
		pointTemp.y = lidarPoint[i].y - minY;
		pointTemp.z = lidarPoint[i].z - minZ;
		lidarPointGravity[i] = pointTemp;
	}
	lidarPoint = lidarPointGravity;
	minmaxVector(lidarPoint, minXYZ, maxXYZ);
}

void AssistClass::gravityPointInverse(std::vector<Point3d> &lidarPoint,std::vector<Point3f> &lidarPointGravity,double *minXYZ,double *maxXYZ)
{
	double minX = minXYZ[0];
	double minY = minXYZ[1];
	double minZ = minXYZ[2];

	for (int i = 0 ; i < lidarPoint.size() ; i++)
	{
		lidarPoint[i].x =lidarPointGravity[i].x+minX;
		lidarPoint[i].y =lidarPointGravity[i].y+minY;
		lidarPoint[i].z =lidarPointGravity[i].z+minZ;
	}
}

void AssistClass::rotateAllLidarPoints(std::vector<Point3d> &lidarPoint)
{
	std::vector<TwoDemensionGrid> lidarPointTwoGrid;
	Mat pointImage;
	double factor = 0.25;
	twoDemensionGridLidarPoint(lidarPoint, lidarPointTwoGrid, pointImage, factor);
	RotatedRect minRect;
	findRotateRect(pointImage,minRect);
	Mat rotatedMatrix;
	computeRotateMatrix(minRect,rotatedMatrix);
	rotateLidarPoint(lidarPoint,rotatedMatrix);
}

void AssistClass::twoDemensionGridLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<TwoDemensionGrid> &lidarPointTwoGrid, Mat &pointImage, double factor)
{
	Point3i featurePoint;
	Point3i leftPoint; 		
	double* minXYZ = new double[3];
	double* maxXYZ = new double[3];

	lidarPointTwoGrid.clear();
	minmaxVector(lidarPoint, minXYZ, maxXYZ);

	int leftX = (int) minXYZ[0];
	int leftY = (int) minXYZ[1];
	int leftZ = (int) minXYZ[2];     

	int rightX = (int)(maxXYZ[0] + 1.9999);
	int rightY = (int) (maxXYZ[1] + 1.9999); 
	int rightZ = (int)(maxXYZ[2] + 1.9999);

	Point3i rightPoint;
	leftPoint.x = leftX; leftPoint.y = leftY; leftPoint.z = leftZ;
	rightPoint.x = rightX; rightPoint.y = rightY; rightPoint.z = rightZ;

	int xDistance = abs(rightX - leftX);
	int yDistance = abs(leftY - rightY);
	int zDistance = abs(rightZ - leftZ);

	int featureImageWidth = (int) (xDistance/factor + 0.9999);
	int featureImageHeight = (int) (yDistance/factor + 0.9999);
	int featureImageElevation = (int)(zDistance/factor + 0.9999);

	featurePoint.x = featureImageWidth;
	featurePoint.y = featureImageHeight;
	featurePoint.z = featureImageElevation;

	lidarPointTwoGrid.resize(featureImageHeight*featureImageWidth);

	for (int i = 0 ; i < (int) lidarPoint.size() ; i++)
	{
		float currentX = lidarPoint[i].x;
		float currentY = lidarPoint[i].y;

		float xDistance = fabs(currentX - leftX);
		float yDistance = fabs(currentY - leftY);

		int xcol = (int)(xDistance/factor);
		int yrow = (int)(yDistance/factor);

		int index = yrow*featureImageWidth + xcol;
		lidarPointTwoGrid[index].gridPointIndex.push_back(i);
	}
	pointImage.create(featureImageHeight,featureImageWidth,CV_8UC1);
	pointImage.setTo(Scalar(0));
	uchar* pointImageData = pointImage.data;

	for (int i = 0 ; i < featureImageHeight ; i++)
	{
		for (int j = 0 ; j < featureImageWidth ; j++)
		{
			int indexForLidar = (featureImageHeight-1-i)*featureImageWidth + j;     //坐标系转换
			if (lidarPointTwoGrid[indexForLidar].gridPointIndex.size())
			{
				*pointImageData = 255;
			}
			pointImageData++;
		}
	}
}

void AssistClass::findRotateRect(Mat &pointImage, RotatedRect &minRect)
{
	uchar *pointImageData = pointImage.data;
	std::vector<Point2f> matPoints;
	for (int i = 0 ; i < pointImage.rows ; i++)
	{
		for (int j = 0 ; j < pointImage.cols ; j++)
		{
			if (*pointImageData)
			{
				matPoints.push_back(Point2f(j,i));   // 注意坐标
			}
			pointImageData++;
		}
	}
	minRect = minAreaRect(matPoints);
	Point2f vertices[4];  
	minRect.points(vertices);  
	for (int i = 0; i < 4; i++)  
		line(pointImage, vertices[i], vertices[(i+1)%4], Scalar(255));

	matPoints.clear();
}

void AssistClass::computeRotateMatrix(RotatedRect &minRect, Mat &currentRotateMatrix)
{
	currentRotateMatrix.create(3,3,CV_64FC1);
	currentRotateMatrix.setTo(Scalar(0));
	double angel = minRect.angle;
	//angel = fabs(angel);
	double radian = (angel/180)*CV_PI;
	double cosAngel = cos(radian);
	double sinAngel = sin(radian);

	currentRotateMatrix.at<double>(0,0) = cosAngel;
	currentRotateMatrix.at<double>(0,1) = sinAngel;
	currentRotateMatrix.at<double>(0,2) = 0;
	currentRotateMatrix.at<double>(1,0) = -sinAngel;
	currentRotateMatrix.at<double>(1,1) = cosAngel;
	currentRotateMatrix.at<double>(1,2) = 0;
	currentRotateMatrix.at<double>(2,0) = 0;
	currentRotateMatrix.at<double>(2,1) = 0;
	currentRotateMatrix.at<double>(2,2) = 1;
}

void AssistClass::rotateLidarPoint(std::vector<Point3d> &preBlockLidarPoint,  Mat &rotatedMatrix)
{
	std::vector<Point3d> preRotatedPoint;
	int pointSize = (int) preBlockLidarPoint.size();

	for (int i = 0 ; i < pointSize ; i++)
	{
		double colx = preBlockLidarPoint[i].x;
		double rowy = preBlockLidarPoint[i].y;
		double elez = preBlockLidarPoint[i].z;
		Point3d projectPoint;
		Mat projectPointMat(1,3,CV_64FC1,Scalar(0));
		projectPointMat.at<double>(0,0) = colx;
		projectPointMat.at<double>(0,1) = rowy;
		projectPointMat.at<double>(0,2) = elez;
		projectPointMat = projectPointMat*rotatedMatrix;
		projectPoint.x = projectPointMat.at<double>(0,0);
		projectPoint.y = projectPointMat.at<double>(0,1);
		projectPoint.z = projectPointMat.at<double>(0,2);
		preRotatedPoint.push_back(projectPoint);
		projectPointMat.release();
	}
	preBlockLidarPoint.clear();
	for (int i = 0 ; i < preRotatedPoint.size() ; i++)
	{
		preBlockLidarPoint.push_back(preRotatedPoint[i]);
	}
	preRotatedPoint.clear();
}

void AssistClass::gridLidarPoint(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, Point3i &featurePoint, Point3i &leftPoint, float factorGrid, int treeTimes, double *minXYZ,double *maxXYZ )
{
	int leftX = (int) minXYZ[0];
	int leftY = (int) minXYZ[1];
	int leftZ = (int) minXYZ[2];     

	int rightX = (int)(maxXYZ[0] + 0.9999);
	int rightY = (int) (maxXYZ[1] + 0.9999); 
	int rightZ = (int)(maxXYZ[2] + 0.9999);

	Point3i rightPoint;
	leftPoint.x = leftX; leftPoint.y = leftY; leftPoint.z = leftZ;
	rightPoint.x = rightX; rightPoint.y = rightY; rightPoint.z = rightZ;

	int xDistance = abs(rightX - leftX);
	int yDistance = abs(leftY - rightY);
	int zDistance = abs(rightZ - leftZ);

	int featureImageWidth = (int) (xDistance/factorGrid);
	int featureImageHeight = (int) (yDistance/factorGrid);
	int featureImageElevation = (int)(zDistance/factorGrid);
	featureImageWidth += (treeTimes - featureImageWidth%treeTimes);
	featureImageHeight += (treeTimes - featureImageHeight%treeTimes);

	if (featureImageElevation%treeTimes)
	{
		featureImageElevation += (treeTimes - featureImageElevation%treeTimes);     //保证八叉树构建顺利
	}
	

	featurePoint.x = featureImageWidth;
	featurePoint.y = featureImageHeight;
	featurePoint.z = featureImageElevation;


	lidarPointGrid.resize(featureImageElevation*featureImageHeight*featureImageWidth);

	for (int i = 0 ; i < (int) lidarPoint.size() ; i++)
	{
		std::vector<int> gridPointIndexTemp;
		std::vector<int> pointMaskTemp;
		float currentX = lidarPoint[i].x;
		float currentY = lidarPoint[i].y;
		float currentZ = lidarPoint[i].z;

		float xDistance = fabs(currentX - leftX);
		float yDistance = fabs(currentY - leftY);
		float zDistance = fabs(currentZ - leftZ);

		int xcol = (int)(xDistance/factorGrid);
		int yrow = (int)(yDistance/factorGrid);
		int zele = (int)(zDistance/factorGrid);

		int index = zele*(featureImageWidth*featureImageHeight)+ yrow*featureImageWidth + xcol;

 		lidarPointGrid[index].gridPointIndex.push_back(i);
	}	
	for (int i = 0 ; i < lidarPointGrid.size() ; i++)
	{
		lidarPointGrid[i].superGridCluster = -1;
	}

}

void AssistClass::constructNeiGrid(std::vector<Grid> &lidarPointGrid, Point3i &featurePoint)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	std::vector<int> eightCol(26);
	std::vector<int> eightRow(26);
	std::vector<int> eightEle(26);
	int eightNeiIndex[26];
	for (int k = 0 ; k < 26 ; k++)
	{
		if (k <= 8 )
			eightEle[k] =  1;
		if (k <= 16 && k >= 9)
			eightEle[k] = 0;
		if (k <= 25 && k >= 17)
			eightEle[k] = - 1;

		if (k == 0 || k == 1 || k == 2 || k == 9 || k == 10 || k == 11 || k == 17 || k == 18 || k == 19)
			eightRow[k] =  - 1;
		if (k == 3 || k == 4 || k == 5 || k == 12 || k == 13 || k == 20 || k == 21 || k == 22)
			eightRow[k] = 0;
		if (k == 6 || k == 7 || k == 8 || k == 14 || k == 15 || k == 16 || k == 23 || k == 24 || k== 25)
			eightRow[k] =  1;

		if (k == 0 || k == 3 || k == 6 || k == 9 || k == 12 || k == 14 || k == 17 || k == 20 || k== 23)
			eightCol[k] =  - 1;
		if (k == 1 || k == 4 || k == 7 || k == 10 || k == 15 || k == 18 || k == 21 || k == 24)
			eightCol[k] = 0;
		if (k == 2 || k == 5 || k == 8 || k == 11 || k == 13 || k == 16 || k == 19 || k == 22 || k== 25)
			eightCol[k] =  1;
	}

	for (int k = 0 ; k < 26 ; k++)
	{
		eightNeiIndex[k] = eightEle[k]*(featureImageWidth*featureImageHeight) + eightRow[k]*featureImageWidth + eightCol[k];
	}

	/**************************求该grid 的26领域*********************/
	for (int i = 0 ; i < featureImageElevation ; i++)
	{
		for (int j = 0 ; j < featureImageHeight ; j++)
		{
			for (int k = 0 ; k < featureImageWidth ; k++)
			{
				std::vector<int> curCol(26);
				std::vector<int> curRow(26);
				std::vector<int> curEle(26);
				
				int gridIndex = i*(featureImageWidth*featureImageHeight) + j*featureImageWidth + k;
				std::vector<int> neiGrid;
				for (int m = 0 ; m < 26 ; m++)
				{
					int curCol = k + eightCol[m];
					int curRow = j + eightRow[m];
					int curEle = i + eightEle[m];
					if (curEle >=0 && curEle < featureImageElevation && curRow >= 0 && curRow < featureImageHeight && curCol >= 0 && curCol < featureImageWidth)
					{
						int curNeiIndex =  curEle*(featureImageWidth*featureImageHeight) + curRow*featureImageWidth + curCol;
						neiGrid.push_back(curNeiIndex);
					}
				}
				lidarPointGrid[gridIndex].neiGridIndex = neiGrid;
			}
		}
	}
}


void AssistClass::normalVector(std::vector<SuperGrid> &lidarSuperGrid, std::vector<GridNormal> &lidaSuperGridNormal)
{
	int superGridSize = (int) lidarSuperGrid.size();
	GridNormal zParameter;
	zParameter.nVectorA = 0;
	zParameter.nVectorB = 0;
	zParameter.nVectorC = 1;

	for (int i = 0 ; i < superGridSize ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		float normalFactor;
		lidaSuperGridNormal[i].isPlane = 0;
		lidaSuperGridNormal[i].superGridIndex = i;
		if (currentSuperGrid.planeParameters.isPlane)
		{
			PlanePa currentPlanePa = currentSuperGrid.planeParameters;
//			normalFactor = sqrtf((currentPlanePa.a*currentPlanePa.a) + (currentPlanePa.b*currentPlanePa.b) + (currentPlanePa.d*currentPlanePa.d));
			lidaSuperGridNormal[i].isPlane = 1;
// 			lidaSuperGridNormal[i].nVectorA = (currentPlanePa.a/normalFactor)*5;
// 			lidaSuperGridNormal[i].nVectorB = (currentPlanePa.b/normalFactor)*5;
// 			lidaSuperGridNormal[i].nVectorC = (currentPlanePa.d/normalFactor)*5;    //整体放大倍数

			lidaSuperGridNormal[i].nVectorA = (currentPlanePa.planeNormal.x)*5;
			lidaSuperGridNormal[i].nVectorB = (currentPlanePa.planeNormal.y)*5;
			lidaSuperGridNormal[i].nVectorC = (currentPlanePa.planeNormal.z)*5;    //整体放大倍数

		}
	}
}

void AssistClass::computeDiffNumber(vector<int> indexClusteTemp, vector<int> &resultIndex)
{
	int number = 0;
	int i,j;
	int m = 0;
	int n = 0;
	int vectorSize = (int) indexClusteTemp.size();
	if (vectorSize == 0)
	{
		return;
	}
	else
	{
		int* clusterTemp = new int[vectorSize];
		for (i = 0 ; i < vectorSize ; i++)
		{
			clusterTemp[i] = -1;
		}
		for (i = 0 ; i < vectorSize ; i++)
		{
			int temp = indexClusteTemp[i];
			if (i == 0)
			{
				clusterTemp[i] = temp;
				m++;
				number++;
			}
			else
			{
				for (j = 0 ; j < m ; j++)
				{
					if (temp == clusterTemp[j])
					{
						clusterTemp[i] = -2;
					}
				}
				if (clusterTemp[i] != -2)
				{
					clusterTemp[i] = temp;
					number++;
				}
				m++;
			}

		}

		for (i = 0 ; i < vectorSize ; i++)
		{
			if (clusterTemp[i] >= 0)
			{
				resultIndex.push_back(clusterTemp[i]);
				n++;
			}
		}
		delete clusterTemp;
	}
}


double AssistClass::computeNormalAngle(cv::Point3d curNormal, cv::Point3d neiNormal)
{
	double angel;
	double ip = curNormal.x*neiNormal.x + curNormal.y*neiNormal.y + curNormal.z*neiNormal.z;
	angel = acos(ip);
	if (ip > 1)
	{
		angel = 0;
	}
	double a = min(fabs(angel - CV_PI), angel);
	return a;
}

double AssistClass::computInnerProduct(cv::Point3d curNormal, cv::Point3d neiNormal)
{
	double ip = curNormal.x*neiNormal.x + curNormal.y*neiNormal.y + curNormal.z*neiNormal.z;
	double innerProdcut = fabs(ip);
	return innerProdcut;
}

double AssistClass::computeDistaceToPlane(cv::Point3d &planeNoraml, cv::Point3d &planePoint, cv::Point3d &neiPoint)
{
	cv::Point3d seedNeiVector;
	seedNeiVector.x = neiPoint.x - planePoint.x;
	seedNeiVector.y = neiPoint.y - planePoint.y;
	seedNeiVector.z = neiPoint.z - planePoint.z;

	double seedNeiVectorLength = sqrtl(seedNeiVector.x*seedNeiVector.x + seedNeiVector.y*seedNeiVector.y + seedNeiVector.z*seedNeiVector.z);
	double innerProduct = (seedNeiVector.x * planeNoraml.x + seedNeiVector.y * planeNoraml.y + seedNeiVector.z * planeNoraml.z);
	double distanceTemp = fabs(innerProduct);
	return distanceTemp;
}


void AssistClass::buildPointPlaneAdjacentGraph(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<SuperGrid> &lidarSuperGrid)
{
	std::vector<int> pointClusterNum(lidarPoint.size(), -1);
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		if (!lidarSuperGrid[i].planeParameters.isPlane)
		{
			continue;
		}
		for (int j = 0 ; j < lidarSuperGrid[i].planePoints.size() ; j++)
		{
			pointClusterNum[lidarSuperGrid[i].planePoints[j]] = i;
		}
	}


	std::vector<std::set<int>> allNeiIndex(lidarSuperGrid.size());
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		int curLabel = i;
		std::vector<int> curCluster = lidarSuperGrid[i].planePoints;
		//std::set<int> tempNeiIndex;
		for (int j = 0 ; j < curCluster.size() ; j++)
		{
			int curIndex = curCluster[j];
			//ANNidxArray curAnnIdxVector = annIdxVector[curIndex];
			std::vector<int> curAnnIdxVector = allPointsNeiIndex[curIndex];
			for (int k = 0 ; k < curAnnIdxVector.size() ; k++)
			{
				int neiPoint = curAnnIdxVector[k];
				int neiLable = pointClusterNum[curAnnIdxVector[k]];
				if (pointClusterNum[curAnnIdxVector[k]] != curLabel && pointClusterNum[curAnnIdxVector[k]] != -1)
				{
					allNeiIndex[i].insert(pointClusterNum[curAnnIdxVector[k]]);
				//	allNeiIndex[i].push_back(pointClusterNum[curAnnIdxVector[k]]);
				}
			}
		}
		
		//std::vector<int> neiIndex;
		for (std::set<int>::iterator it = allNeiIndex[i].begin(); it != allNeiIndex[i].end(); ++it)
		{
			//neiIndex.push_back(*it);
			allNeiIndex[*it].insert(i);
		}

// 		std::vector<int> neiIndex;
// 		for (std::set<int>::iterator it = tempNeiIndex.begin(); it != tempNeiIndex.end(); ++it)
// 		{
// 			neiIndex.push_back(*it);
// 		}
// 		lidarSuperGrid[i].neiGridIndex = neiIndex;
	}

	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		std::vector<int> neiIndex;
		for (std::set<int>::iterator it = allNeiIndex[i].begin(); it != allNeiIndex[i].end(); ++it)
		{
			neiIndex.push_back(*it);
		}
		lidarSuperGrid[i].neiGridIndex = neiIndex;
	}
}


void AssistClass::planeFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters)
{
	int numPoint = allSuperGridPoints.size();
	cv::Matx31d h_mean(0, 0, 0);
	for (int i = 0; i < numPoint; i++)
	{
		h_mean += cv::Matx31d(allSuperGridPoints[i].x, allSuperGridPoints[i].y, allSuperGridPoints[i].z);
	}
	h_mean *= (1.0 / numPoint);

	cv::Matx33d h_cov(0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (int i = 0; i < numPoint; i++)
	{
		cv::Matx31d hi = cv::Matx31d(allSuperGridPoints[i].x, allSuperGridPoints[i].y, allSuperGridPoints[i].z);
		h_cov += (hi - h_mean) * (hi - h_mean).t();
	}
	h_cov *= (1.0 / numPoint);

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen(h_cov, h_cov_evals, h_cov_evectors);

	double t = h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] + (rand() % 10 + 1) * MINVALUE;
	planeParemeters.lamda = h_cov_evals.row(2).val[0] / t;

	cv::Matx31d normalTemp = h_cov_evectors.row(2).t();
	planeParemeters.planeNormal = cv::Point3d(normalTemp.val[0], normalTemp.val[1], normalTemp.val[2]);
	planeParemeters.planePoint = cv::Point3d(h_mean.val[0], h_mean.val[1], h_mean.val[2]);

}


int AssistClass::findNextPoint(double next_x, double next_y, std::vector<std::pair<cv::Point2d, cv::Point2d>> &allEdgeSegments, std::vector<int> &mask)
{
	for (int i = 1 ; i < allEdgeSegments.size() ; i++)
	{
		double first_x = allEdgeSegments[i].first.x;
		double first_y = allEdgeSegments[i].first.y;

		if (next_x == first_x && next_y == first_y && !mask[i] )
		{
			return i;
			break;
		}
	}
	return allEdgeSegments.size();
}


void AssistClass::findMinMax(double *minXY, double *maxXY, std::vector<std::vector<cv::Point3d>> &referencePolyLine, std::vector<std::vector<cv::Point3d>> &detectedPolyLine,
	std::vector<polyLine> &refPolyLines, std::vector<polyLine> &detctPolyLines)
{
	double minX = 100000000;
	double minY = 100000000;

	double maxX = 0;
	double maxY = 0;
	for (int i = 0; i < referencePolyLine.size(); i++)
	{
		double maxZ = 0;
		double minZ = 100000000;
		for (int j = 0; j < referencePolyLine[i].size(); j++)
		{
			if (referencePolyLine[i][j].x < minX)
			{
				minX = referencePolyLine[i][j].x;
			}
			if (referencePolyLine[i][j].y < minY)
			{
				minY = referencePolyLine[i][j].y;
			}
			if (referencePolyLine[i][j].x > maxX)
			{
				maxX = referencePolyLine[i][j].x;
			}
			if (referencePolyLine[i][j].y > maxY)
			{
				maxY = referencePolyLine[i][j].y;
			}

			if (referencePolyLine[i][j].z > maxZ)
			{
				maxZ = referencePolyLine[i][j].z;
			}
			if (referencePolyLine[i][j].z < minZ)
			{
				minZ = referencePolyLine[i][j].z;
			}
		}
		polyLine curLine;
		curLine.polyVetices = referencePolyLine[i];
		curLine.maxZ = maxZ;
		curLine.minZ = minZ;
		curLine.initialIndex = i;
		refPolyLines.push_back(curLine);
	}

	for (int i = 0; i < detectedPolyLine.size(); i++)
	{
		double maxZ = 0;
		double minZ = 100000000;
		for (int j = 0; j < detectedPolyLine[i].size(); j++)
		{
			if (detectedPolyLine[i][j].x < minX)
			{
				minX = detectedPolyLine[i][j].x;
			}
			if (detectedPolyLine[i][j].y < minY)
			{
				minY = detectedPolyLine[i][j].y;
			}
			if (detectedPolyLine[i][j].x > maxX)
			{
				maxX = detectedPolyLine[i][j].x;
			}
			if (detectedPolyLine[i][j].y > maxY)
			{
				maxY = detectedPolyLine[i][j].y;
			}
			if (detectedPolyLine[i][j].z > maxZ)
			{
				maxZ = detectedPolyLine[i][j].z;
			}
			if (detectedPolyLine[i][j].z < minZ)
			{
				minZ = detectedPolyLine[i][j].z;
			}
		}
		polyLine curLine;
		curLine.polyVetices = detectedPolyLine[i];
		curLine.maxZ = maxZ;
		curLine.minZ = minZ;
		curLine.initialIndex = i;
		detctPolyLines.push_back(curLine);
	}

	minXY[0] = minX - 1; minXY[1] = minY - 1; maxXY[0] = maxX + 1; maxXY[1] = maxY + 1;
}

void AssistClass::obtainPlaneEquation(std::vector<polyLine> &detctPolyLines, std::vector<SuperGrid> &detctedPlanes, std::vector<polyLine> &refPolyLines)
{
	for (int i = 0 ; i < detctPolyLines.size() ; i++)
	{
		detctPolyLines[i].a = detctedPlanes[i].planeParameters.planeNormal.x;
		detctPolyLines[i].b = detctedPlanes[i].planeParameters.planeNormal.y;
		detctPolyLines[i].c = detctedPlanes[i].planeParameters.planeNormal.z;
		detctPolyLines[i].d = -(detctPolyLines[i].a *  detctedPlanes[i].planeParameters.planePoint.x + detctPolyLines[i].b *  detctedPlanes[i].planeParameters.planePoint.y + detctPolyLines[i].c *  detctedPlanes[i].planeParameters.planePoint.z);
	}

	for (int i = 0 ; i < refPolyLines.size() ; i++)
	{
		std::vector<cv::Point3d> points;
		std::vector<cv::Point3d> vertices =  refPolyLines[i].polyVetices;
		for (int j = 0; j < vertices.size() - 1; j++)
		{
			points.push_back(vertices[j]);
		}
		PlanePa planePara;
		planeFit(points, planePara);
		refPolyLines[i].a = planePara.planeNormal.x;
		refPolyLines[i].b = planePara.planeNormal.y;
		refPolyLines[i].c = planePara.planeNormal.z;
		refPolyLines[i].d = -(refPolyLines[i].a * planePara.planePoint.x + refPolyLines[i].b * planePara.planePoint.y + refPolyLines[i].c * planePara.planePoint.z);
	}

}

void AssistClass::generateFeatureImage(std::vector<polyLine> refMyPoly, cv::Mat &featureImage, double minX, double minY, double factorGrid)
{
	int imageHeight = featureImage.rows;
	for (int i = 0 ; i < refMyPoly.size() ; i++)
	{
		std::vector<cv::Point3d> curPoly = refMyPoly[i].polyVetices;
		std::vector<cv::Point> imagePoly;
		//std::vector<cv::Point2f> tempPoly;
		for (int j = 0 ; j < curPoly.size() ; j++)
		{
			double currentX = curPoly[j].x;
			double currentY = curPoly[j].y;
			double currentZ = curPoly[j].z;
			double xDistance = currentX - minX;
			double yDistance = currentY - minY;

			int xcol = (int)(xDistance / factorGrid);
			int yrow = (int)(yDistance / factorGrid);
			yrow = imageHeight - yrow;
			imagePoly.push_back(cv::Point(xcol, yrow));

			//tempPoly.push_back(cv::Point2f(xDistance, yDistance));
		}
		//double dis = pointPolygonTest(imagePoly, cv::Point2f(10.0f, 10.0f), true);
		//double dis2 = pointPolygonTest(tempPoly, cv::Point2f(10.0f, 10.0f), true);
		const int rpsize = (int)imagePoly.size();
		const cv::Point* rppoints = &imagePoly[0];
		cv::fillPoly(featureImage, &rppoints, &rpsize, 1, cv::Scalar(i));
	}
}

bool AssistClass::isAdd(int curTemp, std::vector<int> tempVector)
{
	int isAdd = 0;
	if (!tempVector.size())
	{
		return true;
	}
	for (int i = 0 ; i < tempVector.size() ; i++)
	{
		if (curTemp == tempVector[i])
		{
			isAdd++;
			break;
		}
	}
	if (!isAdd)
	{
		return true;//tempVector 中不存在curTemp，则添加
	}
	if (isAdd)
	{
		return false;
	}
}



bool AssistClass::isBelong(int temp, std::vector<int> refVector)
{
	if (!refVector.size())
	{
		return false;
	}
	int isTemp = 0;
	for (int i = 0 ; i < refVector.size() ; i++)
	{
		if (temp == refVector[i])
		{
			isTemp++;
			break;
		}
	}
	if (isTemp)
	{
		return true;
	}
	if (!isTemp)
	{
		return false;
	}
}



void AssistClass::calculateBoundarPrecision(std::vector<cv::Point3d> &gtBoundaryPoints, std::vector<std::vector<cv::Point3d>> &allPlaneEdgePoints)
{
	int pointNum = gtBoundaryPoints.size();
	int kNew = 5;
	double kDis = 0.1;
	ANNpointArray pointData;
	pointData = annAllocPts(pointNum, 3);
	for (int i = 0; i < pointNum; i++)
	{
		pointData[i][0] = gtBoundaryPoints[i].x;
		pointData[i][1] = gtBoundaryPoints[i].y;
		pointData[i][2] = gtBoundaryPoints[i].z;
	}

	std::vector<cv::Point3d> allEdgePoints;
	for (int i = 0; i < allPlaneEdgePoints.size(); i++)
	{
		for (int j = 0; j < allPlaneEdgePoints[i].size(); j++)
		{
			allEdgePoints.push_back(allPlaneEdgePoints[i][j]);
		}
	}

	ANNpointArray ourPointData;
	ourPointData = annAllocPts(allEdgePoints.size(), 3);
	for (int i = 0; i < allEdgePoints.size(); i++)
	{
		ourPointData[i][0] = allEdgePoints[i].x;
		ourPointData[i][1] = allEdgePoints[i].y;
		ourPointData[i][2] = allEdgePoints[i].z;
	}

	std::vector<cv::Point3d> trueBoundaries;  //提取的和gt中都有的边缘

	ANNkd_tree *kdTree = new ANNkd_tree(pointData, pointNum, 3);

	for (int i = 0; i < allEdgePoints.size(); i++)
	{
		cv::Point3d curPoint = allEdgePoints[i];
		ANNidxArray annIdx = new ANNidx[kNew];
		ANNdistArray annDists = new ANNdist[kNew];
		kdTree->annkSearch(ourPointData[i], kNew, annIdx, annDists);

		int isSame = 0;
		for (int j = 0; j < kNew; j++)
		{
			cv::Point3d neiPoints = gtBoundaryPoints[annIdx[j]];
			if (annDists[j] <= kDis)
			{
				isSame++;
				break;
			}
		}
		if (isSame)
		{
			trueBoundaries.push_back(allEdgePoints[i]);
		}
	}

	double bp = (double)trueBoundaries.size() / allEdgePoints.size(); 
	double br = (double)trueBoundaries.size() / gtBoundaryPoints.size();

	cout << "Boundary precision:  " << bp<< "  " <<trueBoundaries.size() << endl;
	cout << "Boundary recall:  " << br << endl;


}
