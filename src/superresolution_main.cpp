/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    superresolution_main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * superresolution_main.cpp
 *
 *  Created on: Feb 29, 2012
 *      Author: steinbrf
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <list>

#include <filesystem/filesystem.h>
#include <superresolution/superresolution.hpp>
//#include <resampling_old.h>
#include <boost/format.hpp>
//#include <convolution.h>
#include <linearoperations/linearoperations.h>

#include <auxiliary/cuda_basic.cuh>

#include <flowlib/flowlib.hpp>


int main(int argc, char *argv[])
{

	std::string sourcedirname = "data";
	std::string resultdirname = "results";

	float superresolution_factor_rescale = 3.0f;
	float superresolution_factor_tv = 10.0f;
	float superresolution_blur = 0.5f;
	float flowBlur = 5.0f;

	bool loadFlows = false;
	bool saveFlows = true;

	Folder folder(sourcedirname);

	std::vector<cv::Mat *> images_original;
	std::vector<cv::Mat *> images_resampled;
	std::vector<cv::Mat *> images_preblurred;
	std::vector<cv::Mat *> images_reconstructed;
	std::vector<float*> flowInputImages;

	std::vector<std::list<Flow> > flows_src_dst;
	std::vector<std::list<Flow> > flows_dst_src;

	bool show_images = true;

	int time_range = 8;

	int computation_device = 0; //0 = CPU, 1 = GPU


	int debug = 0;
	int outer_iterations_1N = 20;
	int outer_iterations_NN = 200;

	for(int arg = 1; arg < argc;){
		std::string parameter = argv[arg++];
		if(parameter == "-m" || parameter == "-cm" || parameter == "-cd"){
			std::string parval = argv[arg++];
			if(parval == "0"){
				computation_device = 0;
			}
			else if(parval == "1"){
				computation_device = 1;
			}
		}
	}


	fprintf(stderr,"\nPGM Files:\n");

	std::vector<std::string> imagenames = folder.getFilesWithEndingVector(".pgm");
//	std::vector<std::string> imagenames = folder.getFilesWithEndingVector(".png");
	for(unsigned i=0;i<imagenames.size();i++){
		fprintf(stderr," %s",imagenames[i].c_str());
	}

	fprintf(stderr,"\nReading Images...");
	for(unsigned int i=0;i<imagenames.size();i++){
		images_original.push_back(new cv::Mat);
		*(images_original.back()) = cv::imread((sourcedirname + "/" + imagenames[i]).c_str(),0);
		images_original.back()->convertTo(*images_original.back(),CV_32FC1);
	} fprintf(stderr,"done.");

	if(!images_original.size()){
		fprintf(stderr,"\nERROR: No Images read!");
		return -1;
	}

	int nx_orig = images_original[0]->cols;
	int ny_orig = images_original[0]->rows;

	for(unsigned int i=0;i<images_original.size();i++){
		if(images_original[i]->cols != nx_orig || images_original[i]->rows != ny_orig){
			fprintf(stderr,"\nERROR: Image Number %i has different Size (%i|%i)!",
					i,images_original[i]->rows,images_original[i]->cols);
			return -1;
		}
	}

	int nx = (int)(nx_orig * superresolution_factor_rescale);
	int ny = (int)(ny_orig * superresolution_factor_rescale);

	fprintf(stderr,"\n Image Dimensions: (%i|%i) => (%i|%i)",nx_orig,ny_orig,nx,ny);

	fprintf(stderr,"\nResampling Images...");
	if(show_images){
		cv::namedWindow("Original Images");
		cv::namedWindow("Resampled Images");
	}
	for(unsigned int i=0;i<images_original.size();i++){
		images_resampled.push_back(new cv::Mat(ny,nx,CV_32FC1));
		images_preblurred.push_back(new cv::Mat(ny,nx,CV_32FC1));

		resampleAreaParallelizableSeparate(
				(float*)images_original[i]->data,(float*)images_resampled[i]->data,
				nx_orig,ny_orig,nx,ny,(float*)images_preblurred[i]->data);

		gaussBlurSeparateMirror((float*)images_resampled.back()->data,(float*)images_preblurred.back()->data,nx,ny,flowBlur,flowBlur);

//		for(int x=0;x<nx*ny;x++) ((float*)(images_preblurred.back()->data))[x] /= 255.0f;
		for(int x=0;x<nx*ny;x++) ((float*)(images_preblurred.back()->data))[x] *= 10.0f;
		flowInputImages.push_back((float*)(images_preblurred.back()->data));
//		cv::imwrite(dirname+"/output_"+imagenames[i]+"_resampled.png",*images_resampled[i]);
		if(show_images){
				cv::imshow("Original Images",(*images_original[i])/255.0f);
				cv::imshow("Resampled Images",(*images_resampled[i])/255.0f);
				cv::imshow("Presmoothed Images",(*images_preblurred[i])/2550.0f);
				cv::waitKey(100);
		}
	}
	if(show_images){
		cv::destroyAllWindows();
		cv::waitKey(100);
	}

	//Creating Flows Filenames
	int flowCounter = 0;
	for(int i=0;(unsigned int)i<images_preblurred.size();i++){
		flows_src_dst.push_back(std::list<Flow>());
		for(int j=i-time_range;j<=i+time_range;j++){
			if(j>=0 && (unsigned int)j<images_preblurred.size()){
				flowCounter++;
				flows_src_dst[i].push_back(Flow());
				flows_src_dst[i].back().nx = nx;
				flows_src_dst[i].back().ny = ny;
				flows_src_dst[i].back().filename =
						sourcedirname + "/flow_" + imagenames[i] + "_" + imagenames[j] +
						(boost::format("_%f")%superresolution_factor_rescale).str()  +
						(boost::format("_%f")%flows_src_dst[i].back().smoothness).str()  +
						".flo";
				flows_src_dst[i].back().src = images_preblurred[i];
				flows_src_dst[i].back().dst = images_preblurred[j];
			}
		}
	}

	if(computation_device == 0){
		for(unsigned int i=0;i<flows_src_dst.size();i++){
			FlowLib *flowlib = NULL;
			fprintf(stderr, " %i",i);
			for(std::list<Flow>::iterator it=flows_src_dst[i].begin();it!=flows_src_dst[i].end();it++){
				it->set(nx,ny);
				if(!loadFlows || !(folder.fileExists(it->filename,sourcedirname+"/"))){
					fprintf(stderr,"\nComputing flow %s",it->filename.c_str());
					if(!flowlib){
						flowlib = new FlowLibCpuSOR(nx,ny);
						flowlib->setLambda(it->smoothness);
						flowlib->setOuterIterations(it->oi);
						flowlib->setInnerIterations(it->ii);
						flowlib->setOverrelaxation(it->overrelaxation);
						flowlib->setStartLevel(it->levels-1);
					}
					flowlib->setInput((float*)it->src->data);
					flowlib->setInput((float*)it->dst->data);
					flowlib->computeFlow();
					flowlib->getOutput(it->u1,it->u2);
					if(saveFlows){
						fprintf(stderr,"\nSaving Flow to File %s",it->filename.c_str());
						it->save_to_file();
						it->saved = true;
					}
				}
				else{
					it->load_from_file();
				}
			}
			delete flowlib;
		}
	}
	else{
		FlowLib *flowlib = NULL;
		for(unsigned int i=0;i<flows_src_dst.size();i++){
			fprintf(stderr, " %i",i);
			for(std::list<Flow>::iterator it=flows_src_dst[i].begin();it!=flows_src_dst[i].end();it++){
				it->set(nx,ny);
				if(!loadFlows || !(folder.fileExists(it->filename,sourcedirname+"/"))){
					fprintf(stderr,"\nComputing flow %s",it->filename.c_str());
					if(!flowlib){
						flowlib = new FlowLibGpuSOR(nx,ny);
						flowlib->setLambda(it->smoothness);
						flowlib->setOuterIterations(it->oi);
						flowlib->setInnerIterations(it->ii);
						flowlib->setOverrelaxation(it->overrelaxation);
						flowlib->setStartLevel(it->levels-1);
					}
					flowlib->setInput((float*)it->src->data);
					flowlib->setInput((float*)it->dst->data);
					flowlib->computeFlow();
					flowlib->getOutput(it->u1,it->u2);
					if(saveFlows){
						fprintf(stderr,"\nSaving Flow to File %s",it->filename.c_str());
						it->save_to_file();
						it->saved = true;
					}
				}
				else{
					it->load_from_file();
				}
			}
		}
		delete flowlib;
	}


	flows_dst_src.resize(flows_src_dst.size(),std::list<Flow>());
	for(int i=0;(unsigned int)i<flows_src_dst.size();i++){
		//		for(std::list<Flow>::iterator it= flows[i].begin();it!=flows[i].end();it++)
		std::list<Flow>::iterator it = flows_src_dst[i].begin();
		int j = (i-time_range<0) ? 0 : (i-time_range);
		while(j<=i+time_range && it!=flows_src_dst[i].end()){
			flows_dst_src[j].push_back(*it);
			j++;
			it++;
		}
	}

	for(unsigned int i=0;i<images_preblurred.size();i++) delete images_preblurred[i];

	if(computation_device == 1){
		size_t total, free;
		if(cuda_memory_available(&total,&free))fprintf(stderr,"\nMemory on GPU: %i of %i",(int)free,(int)total);
	}

	for(int i=0; i<(int)images_resampled.size() ;i++){
		SuperResolution1N *superresolution =
				(computation_device == 1)
				? (SuperResolution1N*)(new SuperResolutionUngerGPU(nx_orig,ny_orig,nx,ny))
				: (SuperResolution1N*)(new SuperResolutionUngerCPU(nx_orig,ny_orig,nx,ny));
		superresolution->_iterations = outer_iterations_NN;
		if(i==10) superresolution->_debug = debug;
		superresolution->_iterations = outer_iterations_1N;
		superresolution->_factor_tv = superresolution_factor_tv;
		superresolution->_blur = superresolution_blur;
		superresolution->_images_original.clear();
		for(int j=i-time_range;j<=i+time_range;j++){
			if(j>=0 && (unsigned int)j<images_original.size()){
				superresolution->_images_original.push_back(images_original[j]);
			}
		}
		superresolution->_flows = flows_dst_src[i];

		fprintf(stderr,"\nComputing...");
		superresolution->compute();
		cv::imwrite(resultdirname+"/sr_"+imagenames[i]+
				(boost::format("_%f")%superresolution_factor_rescale).str() +
				(boost::format("_%f")%superresolution_factor_tv).str() +
				".png",*(superresolution->_result));
		delete superresolution;
	}


	if(computation_device == 1){
		size_t total, free;
		if(cuda_memory_available(&total,&free))fprintf(stderr,"\nMemory on GPU: %i of %i",(int)free,(int)total);
	}

	fprintf(stderr,"\nCleanup...");
	for(unsigned int i=0;i<images_original.size();i++) delete images_original[i];
	for(unsigned int i=0;i<images_resampled.size();i++) delete images_resampled[i];

	fprintf(stderr,"done");

	fprintf(stderr,"\nProgram %s exiting.\n\n",argv[0]);
	return 0;
}



