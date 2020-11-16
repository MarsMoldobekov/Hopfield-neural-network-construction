#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "dirent.h"

using namespace std;
using namespace cv;

void print_syntax() {
    cout << "Syntax:\nNeuralNetwork [input-dir] [output-file]\n";
}

string getFilesExt(string s) {
    size_t i = s.rfind('.', s.length());

    if (i != string::npos) {
        return s.substr(i + 1, s.length() - 1);
    }

    return "";
}

vector<string> fetchFiles(string path) {
    DIR* dir;
    struct dirent* ent;
    vector<string> files;

    dir = opendir(path.c_str());

    while (ent = readdir(dir)) {
        if (getFilesExt(ent->d_name) == "jpg" || getFilesExt(ent->d_name) == "png") {
            files.push_back(path + "\\" + ent->d_name);
        }
    }

    return files;
}

vector<double> mat_to_vector(Mat m, bool normalized = true) {
    assert(m.channels() == 1);
    vector<double> data;

    for (int r = 0; r < m.rows; r++) {
        for (int c = 0; c < m.cols; c++) {
            double val = (double)m.at<uchar>(r, c);

            if (normalized) {
                val /= 255;
            }

            data.push_back(val);
        }
    }

    return data;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_syntax();
        exit(-1);
    }

    vector<string> files = fetchFiles(argv[1]);
    string output_file = argv[2];

    vector<vector<double>> data;

    for (int i = 0; i < files.size(); i++) {
        Mat m = imread(files.at(i), 0);
        data.push_back(mat_to_vector(m));
    }

    ofstream fs(output_file.c_str());

    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data.at(i).size(); j++) {
            fs << data.at(i).at(j);

            if (j != (data.at(i).size() - 1)) {
                fs << ",";
            } else {
                if (i != data.size() - 1) {
                    fs << endl;
                }
            }
        }
    }

    fs.close();
    cout << "Done.\n";

    return(0);
}
