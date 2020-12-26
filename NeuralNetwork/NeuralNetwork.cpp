#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "dirent.h"
#include "neural_network.h"

using namespace std;
using namespace cv;

typedef NeuronNet::state state;

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

Mat change_image_size(const Mat& m, float rescaleFactor) {
    Mat result;
    resize(m, result, Size(), rescaleFactor, rescaleFactor);
    return result;
}

vector<state> mat_to_vector(const Mat& m) {
    vector<state> data;
    vector<uchar> _array;

    Mat new_m = change_image_size(m, 0.5);

    if (new_m.isContinuous()) {
        _array.assign(new_m.data, new_m.data + new_m.total() * new_m.channels());
    }
    else {
        for (int i = 0; i < new_m.rows; i++) {
            _array.insert(_array.end(), new_m.ptr<uchar>(i), new_m.ptr<uchar>(i) + new_m.cols * new_m.channels());
        }
    }

    for_each(_array.begin(), _array.end(),
        [&data](uchar val) {
            data.push_back(NeuronNet::read(val));
        }
    );

    return data;
}

Mat vector_to_mat(const vector<state>& image_v) {
    vector<uchar> image_u;
    for_each(image_v.begin(), image_v.end(), [&image_u](state val) {
        image_u.push_back(NeuronNet::write(val));
    });

    Mat image_m = Mat(image_u, true).reshape(1, 50);
    Mat result = change_image_size(image_m, 2);

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Syntax:\nNeuralNetwork [etallons-dir] [test-image]\n";
        exit(-1);
    }

    vector<string> files = fetchFiles(argv[1]);
    string test_file = argv[2];

    vector<Mat> images;
    list<vector<state>> data;

    for (int i = 0; i < files.size(); i++) {
        Mat m = imread(files.at(i), 0);
        images.push_back(m);
        data.push_back(mat_to_vector(m));
    }
    Mat test_image = imread(test_file, 0);
    vector<state> test = mat_to_vector(test_image);

    NeuronNet net(data);
    auto step = net.recognize(test);
    cout << step << endl;

    Mat recognized_image = vector_to_mat(test);
    imshow("Test Image", test_image);
    imshow("Test Image (Recognized)", recognized_image);

    waitKey(0);
    destroyAllWindows();

    return(0);
}
