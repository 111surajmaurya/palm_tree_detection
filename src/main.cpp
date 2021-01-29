#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;


int img_dim = 64;

HOGDescriptor hog (Size(img_dim,img_dim), Size(16,16), Size(8,8), Size(8,8), 9 ) ;
Mat getHOGFeature(Mat img)
{ 
	Mat gray;
	vector<float> descriptors;
	
	resize(img, img, Size(img_dim,img_dim));
	cvtColor(img, gray, COLOR_BGR2GRAY);
	hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

	Mat feature =Mat(1,1764 ,CV_32FC1);
	memcpy(feature.data,descriptors.data(),descriptors.size()*sizeof(float));
	return feature;
}

void training(string neg_data_path, string pos_data_path, string model_path)
{
	std::vector<int> labels;
    Mat trainingFeature;

    std::vector<cv::String> fn;
    std::vector<cv::String> fn2;
    
    neg_data_path = neg_data_path + "/*";
    glob(neg_data_path, fn, false);
	
	cout << "total neg " << fn.size() << endl;
	int l =0 ;
	for(int i = 0; i< fn.size(); i++)
	{
		cout << i +1<<fn[i] << endl;
		Mat inputTrainImg = imread(fn[i]);
        Mat hogFeature = getHOGFeature(inputTrainImg);            
		trainingFeature.push_back(hogFeature.reshape(1,1));
		labels.push_back(l);
		
	}

	pos_data_path = pos_data_path + "/*";
	glob(pos_data_path, fn2, true);
	l = 1;
	cout << "load pos" <<endl;
	for(int i = 0; i< fn2.size(); i++)
	{
		cout << i +1<<fn2[i] << endl;
		Mat inputTrainImg = imread(fn2[i]);
		Mat hogFeature = getHOGFeature(inputTrainImg);            
		trainingFeature.push_back(hogFeature.reshape(1,1));
		labels.push_back(l);
	}

	cout << "*************" << endl;
	cout << labels.size() << endl;
	cout << trainingFeature.size() << endl;
	 

	Mat trainingDataMat ;
	trainingFeature.convertTo(trainingDataMat , CV_32F);

	cout << "training data " << trainingDataMat.size()<<  trainingDataMat.rows <<endl;
    
    int a = (int)labels.size();
    int lab[a];   
	int totalLabel = labels.size();
	for (int i=0 ; i<(int)labels.size() ;i++ )
	{
		lab[i] = (int)labels.at(i);
	}

    Mat labelsMat(a, 1, CV_32SC1, lab);
    
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 300, 1e-6));

	// Train the SVM with given parameters
	cv::Ptr<cv::ml::TrainData> td =
	    cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

	// or auto train
	svm->trainAuto(td, 20);
	model_path = model_path + "/hog_model.xml";
    svm->save(model_path);
}

void inference(string input_file, string output_file, string model_path)
{
	cout << "Inference" << endl;
	Ptr<ml::SVM> svm;
	svm =  ml::SVM::load(model_path);
	
	
	Mat input = cv::imread(input_file);

	int img_height = input.rows;
	int img_width = input.cols;
	
	Mat resize_img ;
	resize(input, resize_img, Size(400,400));
	float prediction;
	int windows_n_rows = 44;
	int windows_n_cols = 44;
	// Step of each window
	int stride = 8;

	// Mat temp= resize_img.clone();
	Mat out_img = resize_img.clone();
	std::vector<cv::Rect> preds;

	for ( int s = 0; s < 3; s++)
	{
		windows_n_rows += 10;
		windows_n_cols += 10;

		for (int row = 0; row <= resize_img.rows - windows_n_rows; row += stride)
		{
			for (int col = 0; col <= resize_img.cols - windows_n_cols; col += stride)
			{

				Rect windows(col, row, windows_n_rows, windows_n_cols);

				Mat crop = resize_img(windows);
				//resize(crop, crop, Size(64,64));

				Mat hogFeature = getHOGFeature(crop);
				Mat feature;
				feature = hogFeature.reshape(1,1);
				feature.convertTo(feature, CV_32FC1);
				prediction = svm->predict(feature);

				int index = (int)prediction;

				if(index == 1)
				{
					// Rect rr(col+windows_n_cols/2, row+windows_n_rows/2, 2, 2);
					// rectangle(temp, rr, Scalar(0,0,255), 2, 8, 0);
					preds.push_back(windows);
				}
					
				feature.release();
				hogFeature.release();
			}
		}
	}
		
	cv::groupRectangles(preds,1,0.2); 
	
	int box_size = 10;
	for(int i = 0; i<preds.size(); i++)
	{ 
		Rect r = preds[i];
		Point p1, p2;
		p1.x = (r.x + r.width/2 ) -box_size;
		p1.y = (r.y + r.height/2) -box_size;

		p2.x = (p1.x + box_size*2);
		p2.y = (p1.y + box_size*2);

		rectangle(out_img, p1, p2, Scalar(0,0,255), 2, 8, 0);
	}	
	cout << "number of tree " << preds.size() << endl;
	// namedWindow("temp", WINDOW_AUTOSIZE);
	// imshow("temp", temp);
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", out_img);
	waitKey(0);
	imwrite(output_file, out_img);
}


int main(int argc, char* argv[])
{
	bool train_mode = false;
	if(train_mode)
	{
		if(argc < 4) 
		{
	        std::cerr << " Training Usage: " << argv[0] 
	        << " <negative_data_folder_path> <positive_data_folder_path> <output_model_path>" 
	        << std::endl;
	        return 1;
		}
		string neg_data_path, pos_data_path, model_path;
		neg_data_path = argv[1];
		pos_data_path = argv[2];
		model_path = argv[3];
		training(neg_data_path, pos_data_path, model_path);
	}
	else
	{
		if(argc < 4) 
		{
	        std::cerr << " Inference Usage: " << argv[0] 
	        << " <model_path> <input_file> <output_file>" 
	        << std::endl;
	        return 1;
		}
		std::string input_file, output_file, model_path;

		model_path = argv[1];
		input_file = argv[2];
		output_file = argv[3];
		inference(input_file, output_file, model_path);
	}
    return 0;
}
