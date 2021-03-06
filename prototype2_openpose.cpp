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
#include <Windows.h>
#include <strsafe.h>

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


std::string getFileNameFromPath(const std::string path);
std::string getFileNameWithoutExtensionFromPath(const std::string path);
std::string getDirectoryFromPath(const std::string path);
std::string getEarliestCreatedFileNameInDirectory(const std::string& directoryName);
std::string getJsonFromPoseKeyPoints(op::Array<float> poseKeyPoints);
bool outputPoseKeypointsToJson(op::Array<float> poseKeyPoints, const std::string outPath);
int openPoseTutorialPose2(const std::string inImgDirPath, const std::string outDirPath, const std::string archiveImgDirPath,
	const std::string modelDirPath);


int main(int argc, char *argv[])
{
	// Parsing command line flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::string usageMsg = "Usage: prototype2_openpose inImgDirPath outDirPath archiveImgDirPath modelDirPath";
	if (argc != 5)
	{
		op::log("Usage: prototype2_openpose inImgDirPath outDirPath archiveImgDirPath modelDirPath");
		return 1;
	}

	std::string inImgDirPath = std::string(argv[1]);
	std::string outDirPath = std::string(argv[2]);
	std::string archiveImgDirPath = std::string(argv[3]);
	std::string modelDirPath = std::string(argv[4]);

	// TODO: handle error
	CreateDirectory(outDirPath.c_str(), NULL);
	CreateDirectory(archiveImgDirPath.c_str(), NULL);

	int errorCode = openPoseTutorialPose2(inImgDirPath, outDirPath, archiveImgDirPath, modelDirPath);

	return 0;
}

std::string getFileNameFromPath(const std::string path)
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

std::string getFileNameWithoutExtensionFromPath(const std::string path)
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

std::string getDirectoryFromPath(const std::string path)
{
	std::string directory = "";
	const size_t last_slash_idx = path.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		directory = path.substr(0, last_slash_idx);
	}
	return directory;
}

std::string getEarliestCreatedFileNameInDirectory(const std::string& directoryName)
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

std::string getJsonFromPoseKeyPoints(op::Array<float> poseKeyPoints)
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
	}

	if (jsonResult != "")
	{
		jsonResult = jsonResult.substr(0, jsonResult.length() - 1);
		jsonResult += "]}";
	}

	return jsonResult;
}

bool outputPoseKeypointsToJson(op::Array<float> poseKeyPoints, const std::string outPath)
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

int openPoseTutorialPose2(const std::string inImgDirPath, const std::string outDirPath, const std::string archiveImgDirPath,
	const std::string modelDirPath)
{
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

		op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);

		op::CvMatToOpInput cvMatToOpInput{ poseModel };
#if _IsRenderImage
		op::CvMatToOpOutput cvMatToOpOutput;
#endif
		//op::PoseExtractorCaffe poseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start };
		op::PoseExtractorCaffe poseExtractorCaffe{ poseModel, modelDirPath + "/", FLAGS_num_gpu_start };
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




		while (true)
		{
			try
			{


				// get oldest image from input directory
				std::string earliestInImgFileName = getEarliestCreatedFileNameInDirectory(inImgDirPath);
				if (earliestInImgFileName != "")
				{
					const auto timerBegin = std::chrono::high_resolution_clock::now();

					std::string earliestInImgFullPath = inImgDirPath + "\\" + earliestInImgFileName;
					std::string arhiveImgFullPath = archiveImgDirPath + "\\" + earliestInImgFileName;

					// ------------------------- POSE ESTIMATION AND RENDERING -------------------------
					// Step 1 - Read and load image, error if empty (possibly wrong path)
					// Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
					//cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
					cv::Mat inputImage = op::loadImage(earliestInImgFullPath, CV_LOAD_IMAGE_COLOR);
					if (inputImage.empty())
						//op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
						op::error("Could not open or find the image: " + earliestInImgFullPath, __LINE__, __FUNCTION__, __FILE__);
					const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
					// Step 2 - Get desired scale sizes
					std::vector<double> scaleInputToNetInputs;
					std::vector<op::Point<int>> netInputSizes;

					double scaleInputToOutput;
					op::Point<int> outputResolution;
					std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
						= scaleAndSizeExtractor.extract(imageSize);

					// Step 3 - Format input image to OpenPose input and output formats
					const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
#if _IsRenderImage
					auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
#endif
					// Step 4 - Estimate poseKeypoints
					poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
					const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
					const std::string outputFileNameWithoutExtension = getFileNameWithoutExtensionFromPath(earliestInImgFullPath);
					const std::string outputFileFullPath = outDirPath + "\\" + outputFileNameWithoutExtension + "_keypoints.json";
					outputPoseKeypointsToJson(poseKeypoints, outputFileFullPath);
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
					// Return successful message

					// TODO: handle error
					MoveFileEx(earliestInImgFullPath.c_str(), arhiveImgFullPath.c_str(),
						MOVEFILE_REPLACE_EXISTING);
				}
				else
				{
					op::log("No input images in " + inImgDirPath);
				}


			}
			catch (const std::exception& e)
			{
				//op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
				op::log(e.what());
				//return -1;
			}




		}







		return 0;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return -1;
	}
}
