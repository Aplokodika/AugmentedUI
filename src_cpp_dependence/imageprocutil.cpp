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

/*

 """
        for i in range(len(frame)):
            for j in range (len (frame[i])):

                t = []

                for k in range (len (frame[i][j])):
                    t.append(float(frame[i][j][k]))
                    t[k] = t[k] * t[k]

                tSum = sum(t)


                frame[i][j] = [np.uint8( (tVal / tSum)*255 ) for tVal in t]

        """
 */

const uint64_t NO_OF_COLORS_PER_PIXEL = 3;

py::list ImageProcessExtension::exadurateColorByOrder(py::list& imageFrame) {

	uint64_t ySize = py::len(imageFrame);

	uint64_t xSize = py::len(imageFrame[0]);

	py::list result;



	for (uint64_t i = 0; i < ySize; i++) {

		py::list row;


		for (uint64_t j = 0; j < xSize; j++) {
			std::deque <float> temp;
			for (uint64_t k = 0; k < NO_OF_COLORS_PER_PIXEL; k++) {
				temp.push_back((float)py::extract<uint8_t>(imageFrame[i][j][k]));
				temp[k] *= temp[k];
			}

			float sum = 0.0;

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
			.def("exadurateColorByOrder", &ImageProcessExtension::exadurateColorByOrder)
	;

}
