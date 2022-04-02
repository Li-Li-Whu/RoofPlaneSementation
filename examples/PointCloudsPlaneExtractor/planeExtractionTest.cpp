#define _AFXDLL
//#include <afx.h>
#include <string.h>
#include <sys/stat.h>
#include <io.h>

#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "stdio.h"
#include "vector"
#include "iostream"
#include "fstream"
#include "sstream"
#include "math.h"
#include "io.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <Windows.h>

#include "AssistClass.h"
#include "FacadeFootprintExtraction.h"





using namespace std;
using namespace cv;

void filiterSmallPlane(std::vector<cv::Point3d> &lidarPoints, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<cv::Point3d>> &allPlaneEdgePoints)
{
	double minX = 100000000;
	double minY = 100000000;
	double maxX = 0;
	double maxY = 0;
	for (int i = 0; i < lidarPoints.size(); i++)
	{
		if (lidarPoints[i].x < minX)
			minX = lidarPoints[i].x;
		if (lidarPoints[i].y < minY)
			minY = lidarPoints[i].y;
		if (lidarPoints[i].x > maxX)
			maxX = lidarPoints[i].x;
		if (lidarPoints[i].y > maxY)
			maxY = lidarPoints[i].y;
	}
	double factor = 0.5;
	int imageWidth = (int)((maxX - minX) / factor);
	int imageHeight = (int)((maxY - minY) / factor);

	cv::Mat detFeatureImage(imageHeight, imageWidth, CV_32FC1, cv::Scalar(lidarSuperGrid.size() + 1));
	for (int i = 0; i < allPlaneEdgePoints.size(); i++)
	{
		std::vector<cv::Point3d> curPoly = allPlaneEdgePoints[i];
		std::vector<cv::Point> imagePoly;
		for (int j = 0; j < curPoly.size(); j++)
		{
			double currentX = curPoly[j].x;
			double currentY = curPoly[j].y;
			double currentZ = curPoly[j].z;
			double xDistance = currentX - minX;
			double yDistance = currentY - minY;
			int xcol = (int)(xDistance / factor);
			int yrow = (int)(yDistance / factor);
			yrow = imageHeight - yrow;
			imagePoly.push_back(cv::Point(xcol, yrow));
		}
		const int rpsize = (int)imagePoly.size();
		const cv::Point* rppoints = &imagePoly[0];
		cv::fillPoly(detFeatureImage, &rppoints, &rpsize, 1, cv::Scalar(i));
	}

	std::vector<std::vector<cv::Point>> detPlaneCoords(lidarSuperGrid.size());
	float* detFeatureImageData = (float*)detFeatureImage.data;
	for (int i = 0; i < imageHeight; i++)
	{
		for (int j = 0; j < imageWidth; j++)
		{
			int index = i*imageWidth + j;
			int det = (int)detFeatureImageData[index];
			if (det != (lidarSuperGrid.size() + 1))
			{
				detPlaneCoords[det].push_back(cv::Point(j, i));
			}
		}
	}

	double areaThreshold2 = 2.0f;
	int sizeThreshold2 = (int)(areaThreshold2 / factor + 0.5);
	std::vector<int> maskRef(lidarSuperGrid.size(), 0);
	for (int i = 0; i < lidarSuperGrid.size(); i++)
	{
		if (detPlaneCoords[i].size() > sizeThreshold2)
		{
			maskRef[i] = 1;//有效
		}
	}

	std::vector<SuperGrid> tempSuperGrid;
	std::vector<std::vector<cv::Point3d>> tempPlaneEdgePoints;
	for (int i = 0; i < maskRef.size(); i++)
	{
		if (maskRef[i])
		{
			tempSuperGrid.push_back(lidarSuperGrid[i]);
			tempPlaneEdgePoints.push_back(allPlaneEdgePoints[i]);
		}
	}

	lidarSuperGrid = tempSuperGrid;
	allPlaneEdgePoints = tempPlaneEdgePoints;
}

void  fileSearch(vector<string> &allPath,string path)
{

	struct _finddata_t filefind;
	if('\\'==path[path.size()-1])
		path.resize(path.size()-1);
	string curr=path+"\\*.*";
	int  done=0,handle;
	if((handle=_findfirst(curr.c_str(),&filefind))==-1)
	{
		int err=  ::GetLastError();
		return ;
	}
	while(!(done=_findnext(handle,&filefind)))
	{
		if(!strcmp(filefind.name,".."))
			continue;
		curr=path+"\\"+filefind.name;
		//string name = filefind.name;
		char* t= filefind.name;
		strlwr(t);
		//if(strstr(t,name.c_str()))
		allPath.push_back(curr);

		//如果是子目录，再继续找；
		if(_A_SUBDIR==(filefind.attrib & _A_SUBDIR))
			fileSearch(allPath,curr);
	}
	_findclose(handle);
}

////////////////////////用来得到原始数据中标记为点云的数据/////////////////////////////////////////
/*
void main()
{
	AssistClass *assist = new AssistClass;
	std::vector<std::vector<Point3d>> lidarPoints(9);
	FacadeFootprintExtraction *facade = new FacadeFootprintExtraction;
	char* inputLidarRoad1 = "H:\\lidarData\\3DLabelData\\ISPRSData\\Vaihingen3D_Traininig.pts";

	//char* inputLidarRoad1 = "H:\\lidarData\\3DLabelData\\ISPRSData\\Vaihingen3D_EVAL_WITH_REF.pts";
	assist->readXYZPointClouds(inputLidarRoad1, lidarPoints);

	char* outPutRoad = "D:\\building.txt";
	std::vector<std::vector<cv::Point3d>> temp;
	temp.push_back(lidarPoints[5]);
// 	for (int i = 0 ; i < lidarPoints.size() ; i++)
// 	{
// 		temp.push_back(lidarPoints[i]);
// 	}
	assist->writeSegmentePoint(temp, outPutRoad);
}
*/


void main()
{
 	AssistClass assist;

	
	std::vector<Point3d> lidarPoints;

	char* inputLidarRoad1 = "H:\\lidarData\\3DLabelData\\ExperimentDataOurVaihingenBenchmark\\VaihingenRegion1.txt";


	assist.readBuildingXYZPointClouds(inputLidarRoad1, lidarPoints);

	FacadeFootprintExtraction facade;
	std::vector<std::vector<cv::Point3d>> allPlaneEdgePoints;
	std::vector<SuperGrid> detctedPlanes;
	int blockSize = 8;  //  the bolck size for building the tree
	double treeDistanceThreshold = 0.1;  // 0.05 for vaihingen data    0.1 for wuhan data
	double add_DistanceThreshold = 0.2;//2*treeDistanceThreshold; //0.2
	double mergeMSEThreshold = 0.02;     //  0.01 for vaihingen data    0.02 for wuhan data
	double lambda = 5.0;           // The weight between  ditance term and boundary term
	int pointNumThreshold = 15;    // Applied to filter the plane with the point number less than this threshold
	facade.buildingFootPrintExtraction(lidarPoints, detctedPlanes, allPlaneEdgePoints, blockSize, treeDistanceThreshold, mergeMSEThreshold, add_DistanceThreshold, pointNumThreshold, lambda);
	
	
	cout << "Number of detected planes: " << detctedPlanes.size() << endl;

	assist.writeClusters(lidarPoints, detctedPlanes, "D:\\lastPlanes.txt");

	return;



	

}


//////////////////////////指定和global energy一样的初始值做优化/////////////////////////////////////

// void main()
// {
// 	AssistClass assist;
// 	std::vector<Point3d> lidarPoints;
// 	char* inputLidarRoad1 = "H:\\lidarData\\3DLabelData\\ExperimentData\\Wuhan33775+5340\\jietu2\\initial.txt";
// 
// 	std::vector<std::vector<int>> clusters;
// 	assist.readGOPointClouds(inputLidarRoad1, lidarPoints, clusters);
// 
// 
// 	FacadeFootprintExtraction facade;
// 
// 	int blockSize = 8;  //  8 for all datasets
// 	double treeDistanceThreshold = 0.1;  // 0.05 for vaihingen data    0.1 for wuhan data
// 	double mergeMSEThreshold = 0.02;     //  0.01 for vaihingen data    0.02 for wuhan data
// 	double lambda = 5.0;
// 	facade.boundaryRelabellingWithInitialCluster(lidarPoints, clusters, blockSize, lambda);
// }



