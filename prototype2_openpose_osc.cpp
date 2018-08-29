// This code is adapted from 1_extract from image.cpp

// ------------------------- OpenPose Library Tutorial - Pose - Example 1 - Extract from Image -------------------------
// This first example shows the user how to:
// 1. Load an image (`filestream` module)
// 2. Extract the pose of that image (`pose` module)
// 3. Render the pose on a resized copy of the input image (`pose` module)
// 4. Display the rendered pose (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
// 1. `core` module: for the Array<float> class that the `pose` module needs
// 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

// dependencies added by Chris
#include <iostream>
#include <fstream>
//#include <Windows.h>
#include <strsafe.h>





/* Winsock server dependencies */

// Complete Winsock Server Code
// https://docs.microsoft.com/en-us/windows/desktop/winsock/complete-server-code#winsock-server-source-code

#undef UNICODE

#define WIN32_LEAN_AND_MEAN

// https://social.msdn.microsoft.com/Forums/vstudio/en-US/af431084-0b0c-45a7-bdcf-bdf4bf07afcb/help-including-winsock2h-gives-116-errors?forum=vcgeneral
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>

// added by Chris
#include <iostream>
#include <process.h>
#include <string>

using namespace std;

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")

#define DEFAULT_BUFLEN 512
//#define DEFAULT_PORT "27156"
#define HOST_NAME_TO_USE "localhost"
//#define MY_HOST_NAME "127.0.0.1"

/* end of Winsock server dependencies */


// added by Chris
#ifndef _IsRenderImage
#define _IsRenderImage 0
#endif

// See all the available parameter options with the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	" 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	" low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg", "Process the desired image.");
// OpenPose
DEFINE_string(model_pose, "BODY_25", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, "-1x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
	" decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
	" closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
	" any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
	" input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
	" e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	" If you want to change the initial scale, you actually want to multiply the"
	" `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: `part_to_show`,"
	" `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");


/* function declarations */

/* file IO declarations */
string getFileNameFromPath(const string path);
string getFileNameWithoutExtensionFromPath(const string path);
string getDirectoryFromPath(const string path);
string getEarliestCreatedFileNameInDirectory(const string& directoryName);
/* end of file IO declarations */

/* openpose declarations */
string getJsonFromPoseKeyPoints(op::Array<float> poseKeyPoints);
bool outputPoseKeypointsToJson(op::Array<float> poseKeyPoints, const string outPath);
bool initializeOpenPose(op::ScaleAndSizeExtractor *scaleAndSizeExtractor, op::CvMatToOpInput *cvMatToOpInput, op::PoseExtractorCaffe *poseExtractorCaffe,
	string modelDirPath);
/* Important: wrong use of pointer here */
string openPoseGetJsonStrFromImg(string inImgPath,
	op::ScaleAndSizeExtractor *scaleAndSizeExtractor, op::CvMatToOpInput *cvMatToOpInput, op::PoseExtractorCaffe *poseExtractorCaffe,
	string *jsonPoseResult);
struct clientSessionData
{
	SOCKET clientSocket;
	op::ScaleAndSizeExtractor *scaleAndSizeExtractor;
	op::CvMatToOpInput *cvMatToOpInput;
	op::PoseExtractorCaffe *poseExtractorCaffe;
	string *tcpMsgDelimiter;
};
/* end of openpose declarations */

/* Winsock server declarations */
int initializeTcpServer(string hostNameToUse, int port, SOCKET *listenSocket);
void closeDownTcpServer(SOCKET ListenSocket, SOCKET ClientSocket);
unsigned __stdcall ClientSession(void *data);
/* end of Winsock server declarations */

/* end of function declarations */


int main(int argc, char *argv[])
{
	// OpenPose
	// Parsing command line flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);


	// getting modelDirPath(OpenPose) and portToUse(Networking) from command line arguments
	int portToUse;
	string modelDirPath = "";
	string tcpMsgDelimiter = "";
	string usageMsg = "Usage: prototype2_openpose_osc modelDirPath tcpMsgDelimiter portToListen";
	if (argc != 4)
	{
		op::log(usageMsg);
		return 1;
	}
	modelDirPath = string(argv[1]);
	tcpMsgDelimiter = string(argv[2]);
	portToUse = stoi(argv[3]);
	


	/*bool isSuccess;
	op::ScaleAndSizeExtractor *scaleAndSizeExtractor;
	op::CvMatToOpInput *cvMatToOpInput;
	op::PoseExtractorCaffe *poseExtractorCaffe;

	isSuccess = initializeOpenPose(scaleAndSizeExtractor, cvMatToOpInput, poseExtractorCaffe, modelDirPath);
	if (!isSuccess)
	{
		op::log("Error when initializing open pose. Exiting...");
		return 1;
	}*/



	/* open pose initialization */

	op::log("Starting OpenPose demo...", op::Priority::High);


	// ------------------------- INITIALIZATION -------------------------
	// Step 1 - Set logging level
	// - 0 will output all the logging messages
	// - 255 will output nothing
	op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
		__LINE__, __FUNCTION__, __FILE__);
	op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	// Step 2 - Read Google flags (user defined configuration)

	// outputSize
	const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");

	// netInputSize
	const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
	// poseModel
	const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
	// Check no contradictory flags enabled
	if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
		op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
	if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
		op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
			__LINE__, __FUNCTION__, __FILE__);
	// Logging
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	// Step 3 - Initialize all required classes

	op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);


	op::CvMatToOpInput cvMatToOpInput { poseModel };
#if _IsRenderImage
	op::CvMatToOpOutput cvMatToOpOutput;
#endif
	//poseExtractorCaffe = new op::PoseExtractorCaffe { poseModel, FLAGS_model_folder, FLAGS_num_gpu_start };
	op::PoseExtractorCaffe poseExtractorCaffe { poseModel, modelDirPath + "/", FLAGS_num_gpu_start };
#if _IsRenderImage
	op::PoseCpuRenderer poseRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
		(float)FLAGS_alpha_pose };
	op::OpOutputToCvMat opOutputToCvMat;
	std::string frameTitle = "OpenPose Tutorial - Example 1";
	op::FrameDisplayer frameDisplayer{ frameTitle, outputSize };
#endif
	// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
	poseExtractorCaffe.initializationOnThread();
#if _IsRenderImage
	poseRenderer.initializationOnThread();
#endif

	/* end of open pose initialization */



		

	
	int iResult;
	SOCKET ListenSocket = INVALID_SOCKET;
	SOCKET ClientSocket = INVALID_SOCKET;

	
	iResult = initializeTcpServer(HOST_NAME_TO_USE, portToUse, &ListenSocket);
	if (iResult != 0)
	{
		return iResult;
	}

	// TCP Winsock: accept multiple connections/clients
	// https://stackoverflow.com/questions/16686444/function-names-conflict-in-c
	op::log("Port: " + to_string(portToUse) + " is used.");
	op::log("Waiting for incoming socket...");	
	while ((ClientSocket = accept(ListenSocket, NULL, NULL)))
	{
		if (ClientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", WSAGetLastError());
			/*closesocket(ListenSocket);
			WSACleanup();
			return 1;*/
			continue;
		}

		/*struct clientSessionData *data;
		data->clientSocket = ClientSocket;
		data->scaleAndSizeExtractor = &scaleAndSizeExtractor;
		data->cvMatToOpInput = &cvMatToOpInput;
		data->poseExtractorCaffe = &poseExtractorCaffe;
		data->tcpMsgDelimiter = tcpMsgDelimiter;*/

		struct clientSessionData data;
		data.clientSocket = ClientSocket;
		data.scaleAndSizeExtractor = &scaleAndSizeExtractor;
		data.cvMatToOpInput = &cvMatToOpInput;
		data.poseExtractorCaffe = &poseExtractorCaffe;
		data.tcpMsgDelimiter = &tcpMsgDelimiter;

		// Create a new thread for the accepted client (also pass the accepted client socket).
		/*unsigned threadID;
		HANDLE hThread = (HANDLE)_beginthreadex(NULL, 0, &ClientSession, (void*)&data, 0, &threadID);*/

		ClientSession((void*)&data);
	}

	closeDownTcpServer(ListenSocket, ClientSocket);

	return 0;
}


/* file IO implementations */

string getFileNameFromPath(const string path)
{
	std::string fileName = path;
	int pathStrLength = path.length();
	const size_t last_slash_idx = path.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		if (pathStrLength > last_slash_idx)
		{
			fileName = path.substr(last_slash_idx + 1);
		}
		else
		{
			fileName = path.substr(0, pathStrLength - 1);
		}
	}
	return fileName;
}

string getFileNameWithoutExtensionFromPath(const string path)
{
	std::string fileNameWithExtension = getFileNameFromPath(path);
	std::string fileNameWithoutExtension = fileNameWithExtension;
	const size_t dot_idx = fileNameWithExtension.rfind('.');
	if (std::string::npos != dot_idx)
	{
		fileNameWithoutExtension = fileNameWithExtension.substr(0, dot_idx);
	}
	return fileNameWithoutExtension;
}

string getDirectoryFromPath(const string path)
{
	std::string directory = "";
	const size_t last_slash_idx = path.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		directory = path.substr(0, last_slash_idx);
	}
	return directory;
}

string getEarliestCreatedFileNameInDirectory(const string& directoryName)
{
	std::string pattern(directoryName);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;

	if ((hFind = FindFirstFile(pattern.c_str(), &data)) == INVALID_HANDLE_VALUE)
	{
		// Signal error, free memory, (and return an error code?)
		return "";
	}

	FILETIME oldest = { -1U, -1U };
	std::string oldestFile = "";

	do {
		if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue;


		if (CompareFileTime(&(data.ftCreationTime), &oldest) == -1)
			/*if ((data.ftCreationTime.dwHighDateTime < oldest.dwHighDateTime)
			|| (data.ftCreationTime.dwHighDateTime == oldest.dwHighDateTime
			&& data.ftCreationTime.dwLowDateTime < oldest.dwLowDateTime))*/
		{
			oldest = data.ftCreationTime;
			oldestFile = data.cFileName;
		}
	} while (FindNextFile(hFind, &data) != 0);
	FindClose(hFind);

	return oldestFile;
}

/* end of file IO implementations */


/* openpose implementations */

string getJsonFromPoseKeyPoints(op::Array<float> poseKeyPoints)
{
	std::string jsonResult = "{\"version\":1.2,\"people\":[";
	const std::string delimiter = ",";
	const int numOfPoseKeyPointsPerBody = 25 * 3;
	int arrayLength = poseKeyPoints.getVolume();
	for (int i = 0; i < arrayLength; i++)
	{
		int incrementI = (i + 1);

		if (incrementI % numOfPoseKeyPointsPerBody == 1)
		{
			jsonResult += "{\"pose_keypoints_2d\":[";
		}

		jsonResult += std::to_string(poseKeyPoints[i]) + delimiter;

		if (incrementI % numOfPoseKeyPointsPerBody == 0)
		{
			jsonResult = jsonResult.substr(0, jsonResult.length() - 1);
			jsonResult += "],\"face_keypoints_2d\":[],";
			jsonResult += "\"hand_left_keypoints_2d\" : [],";
			jsonResult += "\"hand_right_keypoints_2d\" : [],";
			jsonResult += "\"pose_keypoints_3d\" : [],";
			jsonResult += "\"face_keypoints_3d\" : [],";
			jsonResult += "\"hand_left_keypoints_3d\" : [],",
				jsonResult += "\"hand_right_keypoints_3d\" : []},";
		}

		if (incrementI == arrayLength)
		{

		}
	}

	if (jsonResult != "")
	{
		jsonResult = jsonResult.substr(0, jsonResult.length() - 1);
		jsonResult += "]}";
	}

	return jsonResult;
}

bool outputPoseKeypointsToJson(op::Array<float> poseKeyPoints, const string outPath)
{
	bool isError = false;
	std::ofstream myFile;

	try
	{
		myFile.open(outPath);

		if (myFile.is_open())
		{
			myFile << getJsonFromPoseKeyPoints(poseKeyPoints);
			//myFile << poseKeyPoints;
		}
		else
		{
			isError = false;
		}
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		isError = true;
	}

	myFile.close();

	return !isError;
}

bool initializeOpenPose(op::ScaleAndSizeExtractor *scaleAndSizeExtractor, op::CvMatToOpInput *cvMatToOpInput, op::PoseExtractorCaffe *poseExtractorCaffe,
	string modelDirPath)
{
	bool isSuccess = false;
	
	try
	{
		op::log("Starting OpenPose demo...", op::Priority::High);


		// ------------------------- INITIALIZATION -------------------------
		// Step 1 - Set logging level
		// - 0 will output all the logging messages
		// - 255 will output nothing
		op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 2 - Read Google flags (user defined configuration)

		// outputSize
		const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");

		// netInputSize
		const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
		// poseModel
		const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
		// Check no contradictory flags enabled
		if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
			op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
		if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
			op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
				__LINE__, __FUNCTION__, __FILE__);
		// Logging
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 3 - Initialize all required classes

		scaleAndSizeExtractor = new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);


		cvMatToOpInput = new op::CvMatToOpInput { poseModel };
#if _IsRenderImage
		op::CvMatToOpOutput cvMatToOpOutput;
#endif
		//poseExtractorCaffe = new op::PoseExtractorCaffe { poseModel, FLAGS_model_folder, FLAGS_num_gpu_start };
		poseExtractorCaffe = new op::PoseExtractorCaffe{ poseModel, modelDirPath + "/", FLAGS_num_gpu_start };
#if _IsRenderImage
		op::PoseCpuRenderer poseRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
			(float)FLAGS_alpha_pose };
		op::OpOutputToCvMat opOutputToCvMat;
		std::string frameTitle = "OpenPose Tutorial - Example 1";
		op::FrameDisplayer frameDisplayer{ frameTitle, outputSize };
#endif
		// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
		(*poseExtractorCaffe).initializationOnThread();
#if _IsRenderImage
		poseRenderer.initializationOnThread();
#endif


		isSuccess = true;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}

	return isSuccess;
}

/* Important: wrong use of pointer here */
string openPoseGetJsonStrFromImg(string inImgPath,
	op::ScaleAndSizeExtractor *scaleAndSizeExtractor, op::CvMatToOpInput *cvMatToOpInput, op::PoseExtractorCaffe *poseExtractorCaffe,
	string *jsonPoseResult)
{
	string errMsg = "";

	try
	{
		const auto timerBegin = std::chrono::high_resolution_clock::now();


		// ------------------------- POSE ESTIMATION AND RENDERING -------------------------
		// Step 1 - Read and load image, error if empty (possibly wrong path)
		// Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
		//cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
		cv::Mat inputImage = op::loadImage(inImgPath, CV_LOAD_IMAGE_COLOR);
		if (inputImage.empty())
		{
			//op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
			//op::error("Could not open or find the image: " + inImgPath, __LINE__, __FUNCTION__, __FILE__);

			throw std::runtime_error{ "Could not open or find the image: " + inImgPath };
		}
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
		// Step 2 - Get desired scale sizes
		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;

		double scaleInputToOutput;
		op::Point<int> outputResolution;
		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= (*scaleAndSizeExtractor).extract(imageSize);

		// Step 3 - Format input image to OpenPose input and output formats
		const auto netInputArray = (*cvMatToOpInput).createArray(inputImage, scaleInputToNetInputs, netInputSizes);
#if _IsRenderImage
		auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
#endif
		// Step 4 - Estimate poseKeypoints
		(*poseExtractorCaffe).forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
		const auto poseKeypoints = (*poseExtractorCaffe).getPoseKeypoints();

#if _IsRenderImage
		// Step 5 - Render poseKeypoints
		poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);
		// Step 6 - OpenPose output format to cv::Mat
		auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

		// ------------------------- SHOWING RESULT AND CLOSING -------------------------
		// Show results
		frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
#endif
																 // Measuring total time
		const auto now = std::chrono::high_resolution_clock::now();
		const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - timerBegin).count()
			* 1e-9;
		const auto message = "OpenPose demo successfully finished. Total time: "
			+ std::to_string(totalTimeSec) + " seconds.";
		op::log(message, op::Priority::High);

		// return json containing poseKeypoints
		*jsonPoseResult = getJsonFromPoseKeyPoints(poseKeypoints);
	}
	catch (const exception& e)
	{
		errMsg = e.what();
	}

	return errMsg;
}

/* end of openpose implementations */


/* Winsock server implementations */

int initializeTcpServer(string hostNameToUse, int port, SOCKET *ListenSocket)
{
	// Complete Winsock Server Code
	// https://docs.microsoft.com/en-us/windows/desktop/winsock/complete-server-code#winsock-server-source-code
	WSADATA wsaData;
	int iResult;
	*ListenSocket = INVALID_SOCKET;	
	struct addrinfo *result = NULL;
	struct addrinfo hints;

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed with error: %d\n", iResult);
		return 1;
	}

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	// Resolve the server address and port	
	//iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
	iResult = getaddrinfo(hostNameToUse.c_str(), (to_string(port)).c_str(), &hints, &result);
	if (iResult != 0) {
		printf("getaddrinfo failed with error: %d\n", iResult);
		WSACleanup();
		return 1;
	}

	// Create a SOCKET for connecting to server
	*ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (*ListenSocket == INVALID_SOCKET) {
		printf("socket failed with error: %ld\n", WSAGetLastError());
		freeaddrinfo(result);
		WSACleanup();
		return 1;
	}

	// Setup the TCP listening socket
	// https://stackoverflow.com/questions/16686444/function-names-conflict-in-c
	iResult = ::bind(*ListenSocket, result->ai_addr, (int)result->ai_addrlen);
	if (iResult == SOCKET_ERROR) {
		printf("bind failed with error: %d\n", WSAGetLastError());
		freeaddrinfo(result);
		closesocket(*ListenSocket);
		WSACleanup();
		return 1;
	}

	freeaddrinfo(result);

	iResult = listen(*ListenSocket, SOMAXCONN);
	if (iResult == SOCKET_ERROR) {
		printf("listen failed with error: %d\n", WSAGetLastError());
		closesocket(*ListenSocket);
		WSACleanup();
		return 1;
	}

	return iResult;
}

void closeDownTcpServer(SOCKET ListenSocket, SOCKET ClientSocket)
{
	closesocket(ListenSocket);
	closesocket(ClientSocket);
	WSACleanup();
}

// https://stackoverflow.com/questions/15185380/tcp-winsock-accept-multiple-connections-clients
unsigned __stdcall ClientSession(void *data)
{
	try
	{
		int iResult;
		char recvbuf[DEFAULT_BUFLEN];
		int recvbuflen = DEFAULT_BUFLEN;
		int iSendResult;

		struct clientSessionData *ClientSessionData = (struct clientSessionData *)data;

		SOCKET ClientSocket = ClientSessionData->clientSocket;
		op::ScaleAndSizeExtractor *scaleAndSizeExtractor = ClientSessionData->scaleAndSizeExtractor;
		op::CvMatToOpInput *cvMatToOpInput = ClientSessionData->cvMatToOpInput;
		op::PoseExtractorCaffe *poseExtractorCaffe = ClientSessionData->poseExtractorCaffe;
		string *tcpMsgDelimiter = ClientSessionData->tcpMsgDelimiter;

		string resultantMsg = "";

		// Receive until the peer shuts down the connection
		do {
			iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);

			if (iResult > 0) {
				printf("Bytes received: %d\n", iResult);
				cout << "Bytes converted to str: ";
				cout << recvbuf << endl;

				string recvbufStr = string(recvbuf);
				resultantMsg += recvbufStr;

				// Echo the buffer back to the sender
				iSendResult = send(ClientSocket, recvbuf, iResult, 0);
				if (iSendResult == SOCKET_ERROR) {
					printf("send failed with error: %d\n", WSAGetLastError());
					closesocket(ClientSocket);
					WSACleanup();
					return 1;
				}
				printf("Bytes sent: %d\n", iSendResult);
			}
			else if (iResult == 0)
				printf("Connection closing...\n");
			else {
				printf("recv failed with error: %d\n", WSAGetLastError());
				closesocket(ClientSocket);
				WSACleanup();
				return 1;
			}
		} while (iResult > 0);

		cout << "Received message: ";
		cout << resultantMsg << endl;

		string imgFilePath = resultantMsg.substr(0, resultantMsg.find(*tcpMsgDelimiter));
		string jsonPose;
		string getPoseErrMsg = openPoseGetJsonStrFromImg(imgFilePath,
			scaleAndSizeExtractor, cvMatToOpInput, poseExtractorCaffe,
			&jsonPose);

		if (getPoseErrMsg != "")
		{
			printf(getPoseErrMsg.c_str());
			closesocket(ClientSocket);
			WSACleanup();
			return 1;
		}


		// send something
		const char *sendbuf = jsonPose.c_str();
		iResult = send(ClientSocket, sendbuf, (int)strlen(sendbuf), 0);
		if (iResult == SOCKET_ERROR) {
			printf("send failed with error: %d\n", WSAGetLastError());
			closesocket(ClientSocket);
			WSACleanup();
			return 1;
		}

		cout << "Sent message: ";
		cout << jsonPose << endl;

		// shutdown the connection since we're done
		iResult = shutdown(ClientSocket, SD_SEND);
		if (iResult == SOCKET_ERROR) {
			printf("shutdown failed with error: %d\n", WSAGetLastError());
			closesocket(ClientSocket);
			WSACleanup();
			return 1;
		}
	}
	catch (const exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

/* end of Winsock server implementations */
