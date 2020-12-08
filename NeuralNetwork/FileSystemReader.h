#ifndef FILE_SYSTEM_READER_H
#define FILE_SYSTEM_READER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "dirent.h"

using namespace std;
using namespace cv;

class FileSystemReader {
private:
	DIR* dir;
	vector<string> files;
	vector<Mat> etallons;

	string testFile;
	Mat testImage;

public:
	FileSystemReader(DIR* dir, string testFile) : dir(dir), testFile(testFile) {

	}

	//TODO: realize
	string getFileExt(string s);
	//TODO: change and realize
	void fetchFiles(string path);
};

#endif // FILE_SYSTEM_READER_H
