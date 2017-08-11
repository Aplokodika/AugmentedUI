/*
 * imageprocutil.cpp
 *
 *  Created on: 09-Aug-2017
 *      Author: sreram
 */

#include "imageprocutil.h"

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <boost/cstdint.hpp>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

#include <Python.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <deque>
#include <fstream>
#include <assert.h>
#include <string>

py::dict globals;

// Mockup functions.

/// @brief Converter type that enables automatic conversions between NumPy
///        scalars and C++ types.
template <typename T, NPY_TYPES NumPyScalarType>
struct enable_numpy_scalar_converter
{
  enable_numpy_scalar_converter()
  {
    // Required NumPy call in order to use the NumPy C API within another
    // extension module.
    import_array();

    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<T>());
  }

  static void* convertible(PyObject* object)
  {
    // The object is convertible if all of the following are true:
    // - is a valid object.
    // - is a numpy array scalar.
    // - its descriptor type matches the type for this converter.
    return (
      object &&                                                    // Valid
      PyArray_CheckScalar(object) &&                               // Scalar
      PyArray_DescrFromScalar(object)->type_num == NumPyScalarType // Match
    )
      ? object // The Python object can be converted.
      : NULL;
  }

  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    namespace python = boost::python;
    typedef python::converter::rvalue_from_python_storage<T> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    // Extract the array scalar type directly into the storage.
    PyArray_ScalarAsCtype(object, storage);

    // Set convertible to indicate success.
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


const std::string ImageProcessExtension::__one   	(",1");
const std::string ImageProcessExtension::__zero  	(",0");
const std::string ImageProcessExtension::__comma 	(",");
const std::string ImageProcessExtension::__nextline ("\n");


ImageProcessExtension::ImageProcessExtension ():
								openMode (-1),
								bindType(_DYNAMIC_BIND_),
								fHandle (NULL){

}

ImageProcessExtension::~ImageProcessExtension() {
	close_connection();
}


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

	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {

			uint8_t temp = py::extract<uint8_t>(imageFrame[i][j][0]);

			assert (((temp == py::extract<uint8_t>(imageFrame[i][j][1])) &&
					(temp ==py::extract<uint8_t>(imageFrame[i][j][2]) )) &&
					(temp == 255 || temp == 0 || temp == 1)	&&
					 "Error, this method deals only with binary images. You have given a non-binary image");

			if (temp == 0) {
				lineToBeAdded = lineToBeAdded + __zero;
			} else {
				lineToBeAdded = lineToBeAdded + __one;
			}

		}
	}

	lineToBeAdded = lineToBeAdded + __nextline;

	assert ( (bindType == _DYNAMIC_BIND_  ||
				  bindType == _INSTANT_BIND)  &&
				 "Error, invalid bindType encountered");

	if (bindType == _DYNAMIC_BIND_) {
		fileBuffer.push_back(lineToBeAdded);
		dynamicBindData();
	} else {
		assert (openMode == _APPEND_ && "Error, the open mode must be APPEND, if INSTANT_BIND");
		(*fHandle) << lineToBeAdded;
		fHandle->flush();
	}
}

int ImageProcessExtension::exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return 1;
    }
    return 0;
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


	enable_numpy_scalar_converter <uint8_t, NPY_UBYTE>();

	py::class_<ImageProcessExtension> ("ImageProcessExtension")
			.def_readonly("READ",                 &ImageProcessExtension::_READ_                             )
			.def_readonly("WRITE",                &ImageProcessExtension::_WRITE_                            )
			.def_readonly("APPEND",       		  &ImageProcessExtension::_APPEND_                           )
			.def_readonly("DYNAMIC_BIND", 		  &ImageProcessExtension::_DYNAMIC_BIND_                     )
			.def_readonly("INSTANT_BIND", 		  &ImageProcessExtension::_INSTANT_BIND                      )
			.def("exaggerateColorByOrder",		  &ImageProcessExtension::exaggerate_color_by_order          )
			.def("connectToFile", 		  		  &ImageProcessExtension::connect_to_file                    )
			.def("forceFlushToBuffer",    		  &ImageProcessExtension::force_flush_file_buffer            )
			.def("writeBinaryImageClipToCSVFile", &ImageProcessExtension::write_binary_image_clip_to_CSV_file)
			.def("closeConnection",               &ImageProcessExtension::close_connection)
		;

	py::dict  __globals (py::borrowed(PyEval_GetGlobals()));

	globals = __globals;

	py::exec("import numpy", globals);
}
