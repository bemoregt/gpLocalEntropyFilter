#include "ofApp.h"

using namespace cv;
using namespace ofxCv;

// prototypes ------------------------------------------
static int sub_to_ind(int *coords, int *cumprod, int num_dims);
static void ind_to_sub(int p, int num_dims, const int size[],
                       int *cumprod, int *coords);
void getLocalEntropyImage(cv::Mat &gray, cv::Rect &roi, cv::Mat &entropy);
//--------------------------------------------------------------
void ofApp::setup(){

    aa.allocate(398, 486, OF_IMAGE_COLOR);
    bb.allocate(398, 486, OF_IMAGE_GRAYSCALE);
    aa.load("/Users/kerbal/Desktop/lincon.png");
    
    cv::Mat src;
    src = toCv(aa);
    if (!src.data)
    {
        std::cout << "We have No Image" << std::endl;
        return -1;
    }
    /// Convert to grayscale
    cvtColor(src, src, cv::COLOR_BGR2GRAY);
    
    //Calculate Entropy Filter
    cv::Rect roi(0, 0, src.cols, src.rows);
    cv::Mat dst(src.rows, src.cols, CV_32F);
    getLocalEntropyImage(src, roi, dst);
    //cv::normalize(dst, dst, 255, 0, cv::NORM_MINMAX);
    cv::Mat entropy;
    cv::normalize(dst, entropy, 255, 0, cv::NORM_MINMAX, CV_8U);
    
    //dst.convertTo(entropy, CV_8U);
    
    //cv::imshow("Original", src);
    //cv::imshow("Entropy Filter", dst);
    //cv::imshow("Entropy Filter2", entropy);
    toOf(src, aa);
    toOf(entropy, bb);
    aa.update();
    bb.update();
    
    ofBackground(50);
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

    ofSetColor(255);
    aa.draw(0, 0, 398, 486);
    if(bb.isAllocated()){
        bb.draw(398, 0, 398, 486);
    }
    //
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
// funtions --------------------------------------------
static int sub_to_ind(int *coords, int *cumprod, int num_dims)
{
    int index = 0;
    int k;
    
    assert(coords != NULL);
    assert(cumprod != NULL);
    assert(num_dims > 0);
    
    for (k = 0; k < num_dims; k++)
    {
        index += coords[k] * cumprod[k];
    }
    
    return index;
}
//------------------------------------------------------
static void ind_to_sub(int p, int num_dims, const int size[],
                       int *cumprod, int *coords)
{
    int j;
    
    assert(num_dims > 0);
    assert(coords != NULL);
    assert(cumprod != NULL);
    
    for (j = num_dims - 1; j >= 0; j--)
    {
        coords[j] = p / cumprod[j];
        p = p % cumprod[j];
    }
}
//---------------------------------------------------------
void getLocalEntropyImage(cv::Mat &gray, cv::Rect &roi, cv::Mat &entropy)
{
    using namespace cv;
    clock_t func_begin, func_end;
    func_begin = clock();
    //1.define nerghbood model,here it's 9*9
    int neighbood_dim = 2;
    int neighbood_size[] = {9, 9};
    
    //2.Pad gray_src
    Mat gray_src_mat(gray);
    Mat pad_mat;
    int left = (neighbood_size[0] - 1) / 2;
    int right = left;
    int top = (neighbood_size[1] - 1) / 2;
    int bottom = top;
    copyMakeBorder(gray_src_mat, pad_mat, top, bottom, left, right, BORDER_REPLICATE, 0);
    Mat *pad_src = &pad_mat;
    roi = cv::Rect(roi.x + top, roi.y + left, roi.width, roi.height);
    
    //3.initial neighbood object,reference to Matlab build-in neighbood object system
    //        int element_num = roi_rect.area();
    //here,implement a histogram by ourself ,each bin calcalate gray value frequence
    int hist_count[256] = {0};
    int neighbood_num = 1;
    for (int i = 0; i < neighbood_dim; i++)
        neighbood_num *= neighbood_size[i];
    
    //neighbood_corrds_array is a neighbors_num-by-neighbood_dim array containing relative offsets
    int *neighbood_corrds_array = (int *)malloc(sizeof(int)*neighbood_num * neighbood_dim);
    //Contains the cumulative product of the image_size array;used in the sub_to_ind and ind_to_sub calculations.
    int *cumprod = (int *)malloc(neighbood_dim * sizeof(*cumprod));
    cumprod[0] = 1;
    for (int i = 1; i < neighbood_dim; i++)
        cumprod[i] = cumprod[i - 1] * neighbood_size[i - 1];
    int *image_cumprod = (int*)malloc(2 * sizeof(*image_cumprod));
    image_cumprod[0] = 1;
    image_cumprod[1] = pad_src->cols;
    //initialize neighbood_corrds_array
    int p;
    int q;
    int *coords;
    for (p = 0; p < neighbood_num; p++){
        coords = neighbood_corrds_array + p * neighbood_dim;
        ind_to_sub(p, neighbood_dim, neighbood_size, cumprod, coords);
        for (q = 0; q < neighbood_dim; q++)
            coords[q] -= (neighbood_size[q] - 1) / 2;
    }
    //initlalize neighbood_offset in use of neighbood_corrds_array
    int *neighbood_offset = (int *)malloc(sizeof(int) * neighbood_num);
    int *elem;
    for (int i = 0; i < neighbood_num; i++){
        elem = neighbood_corrds_array + i * neighbood_dim;
        neighbood_offset[i] = sub_to_ind(elem, image_cumprod, 2);
    }
    
    //4.calculate entropy for pixel
    uchar *array = (uchar *)pad_src->data;
    //here,use entroy_table to avoid frequency log function which cost losts of time
    float entroy_table[82];
    const float log2 = log(2.0f);
    entroy_table[0] = 0.0;
    float frequency = 0;
    for (int i = 1; i < 82; i++){
        frequency = (float)i / 81;
        entroy_table[i] = frequency * (log(frequency) / log2);
    }
    int neighbood_index;
    //        int max_index=pad_src->cols*pad_src->rows;
    float e;
    int current_index = 0;
    int current_index_in_origin = 0;
    for (int y = roi.y; y < roi.height; y++){
        current_index = y * pad_src->cols;
        current_index_in_origin = (y - 4) * gray.cols;
        for (int x = roi.x; x < roi.width; x++, current_index++, current_index_in_origin++) {
            for (int j = 0; j<neighbood_num; j++) {
                neighbood_index = current_index + neighbood_offset[j];
                hist_count[array[neighbood_index]]++;
            }
            //get entropy
            e = 0;
            for (int k = 0; k < 256; k++){
                if (hist_count[k] != 0){
                    //                                        int frequency=hist_count[k];
                    e -= entroy_table[hist_count[k]];
                    hist_count[k] = 0;
                }
            }
            ((float *)entropy.data)[current_index_in_origin] = e;
        }
    }
    free(neighbood_offset);
    free(image_cumprod);
    free(cumprod);
    free(neighbood_corrds_array);
    
    func_end = clock();
    double func_time = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
    std::cout << "func time" << func_time << std::endl;
}
