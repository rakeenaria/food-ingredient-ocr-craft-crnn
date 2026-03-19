#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const char *keys =
    "{ help     h  | | Print help message. }"
    "{ modelPath   |    | Path to a binary .onnx file contains trained text recognition model.}"
    "{ imgPath     |    | test image path}";

const std::string vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz";

std::string decodeRec(Mat prediction);

void PrintMat(Mat A)
{ 
    cout<<"size of NCHW = ["<<A.size[0]<<" x "<<A.size[1]<<" x " <<A.size[2]<<"]"<<endl;
    cout<<"h = "<<A.rows<<", w = "<<A.cols<<endl;
    for(int i=0;i<A.rows;i++)
    {
        for(int j=0;j<A.cols;j++)
        cout<<A.at<Vec3b>(i,j)<<' ';
        // cout<<A.at<uchar>(i,j)<<' ';
        cout<<endl;
    }
    cout<<endl;
}

Mat keepImageRatioWithPad(Mat image, int bolbW, int blobH)
{ 
    // imgH = 32, imgW = 100
    int imgW = image.size().width;
    int imgH = image.size().height;

    int resizedW = 0;
    float aspectRatio = float(imgW)/float(imgH);

    // Mat resizedImage(Size(100,32), CV_8UC3, Scalar(0));
    if(ceil(aspectRatio * blobH) > bolbW )
    {
        resizedW = bolbW;
    }
    else
    {
        resizedW = ceil(aspectRatio * blobH);
    }

    resize(image, image, Size(resizedW, blobH));

    if(resizedW < bolbW){
        transpose(image, image);
        Mat colData = image.row(resizedW-1);
        for(int i = resizedW; i< bolbW; i++){
            image.push_back(colData);
        }
        transpose(image, image);
    }
    cout<<"resized imag size "<<image.size()<<endl;
    return image;
}

int main(int argc, char **argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string modelPath = parser.get<String>("modelPath");
    string imgPath = parser.get<String>("imgPath");
    static const std::string kWinCrop = "show crop image";

    namedWindow(kWinCrop, WINDOW_AUTOSIZE);

    dnn::Net net;
    try
    {
        net = dnn::readNet(modelPath);
        cout<<"model load sucessuful"<<endl;
    }
    catch (cv::Exception &ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        if (net.empty())
        {
            std::cout << "Can't load the network by using the flowing files:" << std::endl;
            std::cout << "modelPath: " << modelPath << std::endl;
            return 1;
        }
    }

    Mat pred;
    Mat img = imread(imgPath, IMREAD_GRAYSCALE);

    // if keep image ratio
    // img = keepImageRatioWithPad(img, 100, 32);
    // imshow("keep ratio ", img);
    // waitKey(200);


    double scale = 1.0/255.0;
    Mat blobImg = dnn::blobFromImage(img,scale, Size(100,32),Scalar(),true);  // NCHW = 
    blobImg -= 0.5;
    blobImg /= 0.5;

    const string input_name = string("input");
    net.setInput(blobImg, input_name);
    pred = net.forward();
    string decodeSeq = decodeRec(pred);

    cout<<" text recog output is :"<<decodeSeq<<endl; 
	return 0;
}
 
std::string decodeRec(Mat prediction)
{
    std::string decodeSeq = "";
    bool ctcFlag = true;
    for (int i = 0; i < prediction.size[0]; i++) {
        int maxLoc = 0;
        float maxScore = prediction.at<float>(i, 0);
        for (uint j = 0; j < vocabulary.length() + 1; j++) {
            float score = prediction.at<float>(i, j);
            if (maxScore < score) {
                maxScore = score;
                maxLoc = j;
            }
        }
        if (maxLoc > 0) {
            char currentChar = vocabulary[maxLoc - 1];
            if (currentChar != decodeSeq.back() || ctcFlag) {
                decodeSeq += currentChar;
                ctcFlag = false;
            }
        } else {
            ctcFlag = true;
        }
    }

    return decodeSeq;
}
