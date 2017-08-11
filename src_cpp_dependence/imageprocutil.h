/*
 * imageprocutil.h
 *
 *  Created on: 09-Aug-2017
 *      Author: sreram
 */

#ifndef IMAGEPROCUTIL_H_
#define IMAGEPROCUTIL_H_

#include <boost/python.hpp>
#include <Python.h>
namespace py = boost::python;

struct ImageProcessExtension {

	py::list exaggerateColorByOrder (py::object& pImageFrame, py::object porder);



};


#endif /* IMAGEPROCUTIL_H_ */
