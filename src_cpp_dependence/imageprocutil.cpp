/*
 * imageprocutil.cpp
 *
 *  Created on: 09-Aug-2017
 *      Author: sreram
 */

#include "imageprocutil.h"

#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <boost/cstdint.hpp>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

#include <iostream>
#include <vector>
#include <stdint.h>
#include <deque>
#include <fstream>
#include <assert.h>
#include <string>
#include <cstdlib>
#include <sstream>

py::dict globals;


template <typename T, NPY_TYPES NumpyType>
struct configure_numpy_data_convertion
{
	configure_numpy_data_convertion()
  {

    import_array();

    py::converter::registry::push_back( &isconvertible, &perform_convertion,
    py::type_id<T>());
  }

  static void* isconvertible(PyObject* var)
  {
    if ( var && PyArray_CheckScalar(var) &&
    		PyArray_DescrFromScalar(var)->type_num == NumpyType)
    	return var;
      else
    	  return NULL;
  }

  static void perform_convertion( PyObject* var,
		  py::converter::rvalue_from_python_stage1_data* data)
  {

    typedef py::converter::rvalue_from_python_storage<T> DestinationType;
    void* storage = reinterpret_cast<DestinationType*>(data)->storage.bytes;
    PyArray_ScalarAsCtype(var, storage);

    // signify successful conversion.
    data->convertible = storage;
  }
};



const uint64_t NO_OF_COLORS_PER_PIXEL = 3;

const int ImageProcessExtension::_READ_   = 0;
const int ImageProcessExtension::_WRITE_  = 1;
const int ImageProcessExtension::_APPEND_ = 2;

const int ImageProcessExtension::_DYNAMIC_BIND_ = 0;
const int ImageProcessExtension::_INSTANT_BIND  = 1;

const int ImageProcessExtension::_MAX_THRESHOLD_DYNAMIC_ = 1000;


const std::string ImageProcessExtension::__one   	 ("1");
const std::string ImageProcessExtension::__zero  	 ("0");
const std::string ImageProcessExtension::__comma 	 (",");
const std::string ImageProcessExtension::__nextline  ("\n");
const std::string ImageProcessExtension::_blankspace (" ");


ImageProcessExtension::ImageProcessExtension ():
								readPos(0),
								openMode (-1),
								bindType(_DYNAMIC_BIND_),
								fHandle (NULL){

}

ImageProcessExtension::~ImageProcessExtension() {
	close_connection();
}




void ImageProcessExtension::resetReadPos () { readPos = 0; }

void ImageProcessExtension::setReadPos(py::object pos) { readPos = py::extract <int> (pos);}


void ImageProcessExtension::close_connection () {
	if (fHandle != NULL) {
		fHandle->close();
		delete fHandle;
	}
	fHandle = NULL;
}

py::list ImageProcessExtension::convert_image_frame_to_list (py::object& pImageFrame) {

	globals["imageFrame"] = pImageFrame;
	py::exec("imageFrame = list(imageFrame)", globals);
	return py::list(globals["imageFrame"]);
}


bool ImageProcessExtension::isnumber (std::string str) {
	for (int i = 0; i < str.size(); i++) {
		if (str[i] == '\n') {

			if (i == 0) {
				return false;
			} else {
				return true;
			}

		}
		if ( std::isdigit(str[i]) != true){
			return false;
		}
	}

	return true;

}


int ImageProcessExtension::atoi (std::string val) {
	int result;

	std::stringstream sstream (val);
	sstream >> result;

	return result;

}

std::string ImageProcessExtension::itoa(int num) {

	char* buffer = new char [12];
	sprintf(buffer,"%d", num);
	std::string retval(buffer);
	delete [] buffer;
	return retval;
}


void ImageProcessExtension::force_flush_file_buffer () {

	if ( openMode != _APPEND_ || bindType != _DYNAMIC_BIND_)
		return;

	fHandle->seekp(std::ios_base::end);

	while (fileBuffer.size()) {
		(*fHandle) << fileBuffer[0];
		fileBuffer.pop_front(); // acts like a queue. Pushed back, popped front.
	}
}


void ImageProcessExtension::dynamicBindData() {

	if (fileBuffer.size() < _MAX_THRESHOLD_DYNAMIC_)
		return;

	force_flush_file_buffer ();
}

/*	@ImageProcessExtension::get_comma_separated_strings
 *
 * 	For each "comma symbol or blank-space or '\n' symbol" read by this method,
 * 	the symbols read by the stream is pushed into the symbol list.
 * 	pLine contains a string of numbers separated by comma (csv format).
 *
 */
std::deque <std::string> ImageProcessExtension::get_comma_separated_strings (std::string pLine) {

	std::deque <std::string> result; // returns the list of symbols separated by comma.
	std::ostringstream strStream;	 // stores the symbol read from the string pLine for each iteration

	for (uint64_t i = 0; i  < pLine.size(); i++) {

		// separates for 'comma', 'nextline' or 'blankspace'.
		if (  (pLine[i] == __comma    [0])	||
			  (pLine[i] == __nextline [0])  ||
			  (pLine[i] == _blankspace[0])) {

			// Adds the parsed symbol once 'comma' or 'newline' is read.
		    // (Also, the stream must not be empty).
			if ( strStream.rdbuf()->in_avail() != 0) {
				result.push_back(strStream.str());
				strStream.clear();
			}
		} else {
			strStream << pLine[i];
		}

	}

	return result;
}


py::list ImageProcessExtension
::read_binary_image_clip_from_CSV_file (py::object& representationOfValue1,
										py::object& representationOfValue0) {

	assert ( (openMode == _READ_) &&
			 "Error, the file should have been opened in READ mode before calling this method");


	const int VALUE_1 = py::extract <int> (representationOfValue1);
	const int VALUE_0 = py::extract <int> (representationOfValue0);

	py::list result;
	std::string sizeParameters;
	std::string lineReadFromFile;
	std::deque <std::string> symbolList;

	fHandle->seekg(this->readPos, std::ios_base::beg);

	(*fHandle) >> sizeParameters;

	symbolList = this->get_comma_separated_strings(sizeParameters);

	assert ((symbolList.size() == 2) 		  &&
			(isnumber(symbolList[0]) == true) &&
			(isnumber(symbolList[1]) == true) &&
			"Error, the size of the segment-header is either less than 2 or the parameters obtained are not numbers");

	int ySize  = this->atoi(symbolList[0]);
	int xSize  = this->atoi(symbolList[1]);

	assert ((symbolList.size() == xSize * ySize ) &&
			"Error, there is a mismatch in the the expected number of pixels and the given number of pixels");

	(*fHandle) >> lineReadFromFile;
	std::deque <std::string> _tempSymbolList = this->get_comma_separated_strings(lineReadFromFile);

	py::list image;

	for (int i = 0, k = 0; i < ySize; i++) {

		py::list row;
		for (int j = 0; j < xSize; j++, k++) {
			py::list pixel;
			int curValue = ((atoi(_tempSymbolList[k]) == 1) ? VALUE_1: VALUE_0 ) ;
			for (int k = 0; k < NO_OF_COLORS_PER_PIXEL; k++) {
				pixel.append((uint8_t)curValue);
			}
			row.append(pixel);
		}
		image.append(row);
	}

	return image;
}

void ImageProcessExtension::write_binary_image_clip_to_CSV_file (py::object& pImageFrame) {

	assert ( (openMode == _WRITE_ ||
			  openMode == _APPEND_ ) &&
			"Error, the file should have been opened in WRITE or APPEND mode before calling this method");

	py::list imageFrame  = convert_image_frame_to_list (pImageFrame);

	int ySize = py::len(imageFrame);
	int xSize = py::len(imageFrame[0]);

	std::string lineToBeAdded;

	lineToBeAdded = std::string(itoa(ySize)) + __comma +
			 	 	std::string(itoa(xSize));

	std::ostringstream strStream;

	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {

			uint8_t temp = py::extract<uint8_t>(imageFrame[i][j][0]);

			assert (((temp == py::extract<uint8_t>(imageFrame[i][j][1])) &&
					(temp ==py::extract<uint8_t>(imageFrame[i][j][2]) )) &&
					(temp == 255 || temp == 0 || temp == 1)	&&
					 "Error, this method deals only with binary images. You have given a non-binary image");

			if (temp == 0) {

				strStream << __comma << __zero;

				// lineToBeAdded = lineToBeAdded + __comma + __zero;
			} else {

				strStream << __comma <<__one;

				//lineToBeAdded = lineToBeAdded + __comma + __one;
			}

		}
	}


	strStream << __nextline;
	// lineToBeAdded = lineToBeAdded + __nextline;

	assert ( (bindType == _DYNAMIC_BIND_  ||
				  bindType == _INSTANT_BIND)  &&
				 "Error, invalid bindType encountered");

	if (bindType == _DYNAMIC_BIND_) {
		//fileBuffer.push_back(lineToBeAdded);

		fileBuffer.push_back(strStream.str());

		dynamicBindData();
	} else {
		assert (openMode == _APPEND_ && "Error, the open mode must be APPEND, if INSTANT_BIND");
		//(*fHandle) << lineToBeAdded;

		(*fHandle) << strStream.str();
		fHandle->flush();
	}
}

int ImageProcessExtension::exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return true;
    }
    return false;
}

void ImageProcessExtension::connect_to_file (py::object& pFName,
		py::object& pOpenMode, py::object& pBindType) {

	force_flush_file_buffer();
	close_connection();

	std::string fName  = py::extract<std::string> (pFName);
	openMode = py::extract <int> (pOpenMode);
	bindType = py::extract <int> (pBindType);

	assert ( (openMode == _READ_   ||
			  openMode == _WRITE_  ||
			  openMode == _APPEND_  ) && "Error, pOpenMode can only take the values 0(READ), 1(WRITE) or 2(APPEND)");

	std::ios_base::openmode opMode =  (openMode == _READ_)  ? std::ios_base::in:
			  	  	  	   ((openMode == _WRITE_) ? std::ios_base::out:
			  	  	  			   	   	   	   	    std::ios_base::app);
	if (!exists(fName.c_str())){
		std::fstream newF(fName.c_str(), std::ios_base::out);
		newF.close();
	}

	fHandle = new std::fstream(fName.c_str(), opMode);

}



py::list ImageProcessExtension::exaggerate_color_by_order(py::object& pImageFrame, py::object porder) {

	int order = py::extract<int> (porder);

	py::list imageFrame =  convert_image_frame_to_list(pImageFrame);

	uint64_t ySize = py::len(imageFrame);
	uint64_t xSize = py::len(imageFrame[0]);
	py::list result;

	double prod, sum;

	for (uint64_t i = 0; i < ySize; i++) {

		py::list row;

		for (uint64_t j = 0; j < xSize; j++) {

			std::deque <double> temp;

			for (uint64_t k = 0; k < NO_OF_COLORS_PER_PIXEL; k++) {
				temp.push_back((double)py::extract<uint8_t>(imageFrame[i][j][k]));
				prod = 1.0;
				for (uint8_t i = 1; i < order; i++) {
					prod *= temp[k];
				}
			    temp[k] *= prod;
			}

			sum = 0.0;

			for (uint64_t k= 0; k < 3; k++) {
				sum += temp[k];
			}

			py::list pixel;

			for (uint64_t k = 0; k < NO_OF_COLORS_PER_PIXEL; k++ ) {
				pixel.append((uint8_t)((temp[k]/sum)*255));
			}

			row.append(pixel);

		}


		result.append(row);

	}

	return result;

}


BOOST_PYTHON_MODULE(ImageProcessExtension)
{

	import_array();


	configure_numpy_data_convertion <uint8_t, NPY_UBYTE>();

	py::class_<ImageProcessExtension> ("ImageProcessExtension")
			.def_readonly("READ",                  &ImageProcessExtension::_READ_                              )
			.def_readonly("WRITE",                 &ImageProcessExtension::_WRITE_                             )
			.def_readonly("APPEND",       		   &ImageProcessExtension::_APPEND_                            )
			.def_readonly("DYNAMIC_BIND", 		   &ImageProcessExtension::_DYNAMIC_BIND_                      )
			.def_readonly("INSTANT_BIND", 		   &ImageProcessExtension::_INSTANT_BIND                       )
			.def("exaggerateColorByOrder",		   &ImageProcessExtension::exaggerate_color_by_order           )
			.def("connectToFile", 		  		   &ImageProcessExtension::connect_to_file                     )
			.def("forceFlushToBuffer",    		   &ImageProcessExtension::force_flush_file_buffer             )
			.def("writeBinaryImageClipToCSVFile",  &ImageProcessExtension::write_binary_image_clip_to_CSV_file )
			.def("closeConnection",                &ImageProcessExtension::close_connection					   )
			.def("readBinaryImageClipFromCSVfile", &ImageProcessExtension::read_binary_image_clip_from_CSV_file)
		;

	py::dict  __globals (py::borrowed(PyEval_GetGlobals()));

	globals = __globals;

	py::exec("import numpy", globals);
}
