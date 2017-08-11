/*
 * imageprocutil.h
 *
 *  Created on: 09-Aug-2017
 *      Author: sreram
 */

#ifndef IMAGEPROCUTIL_H_
#define IMAGEPROCUTIL_H_

#include <fstream>
#include <deque>
#include <stdint.h>
#include <string>

#include <boost/python.hpp>
#include <Python.h>

namespace py = boost::python;

struct ImageProcessExtension {


	int openMode;

	int bindType;

	std::fstream *fHandle;

	static const int _READ_  ;
	static const int _WRITE_ ;
	static const int _APPEND_;


	static const int _DYNAMIC_BIND_;
	static const int _INSTANT_BIND;


	static const int _MAX_THRESHOLD_DYNAMIC_;

	ImageProcessExtension ();

	~ImageProcessExtension();


	void force_flush_file_buffer ();

	py::list exaggerate_color_by_order (py::object& pImageFrame, py::object porder);

	void connect_to_file (py::object& pFName, py::object& pOpenMode,
			py::object& pBindType);

	void close_connection ();

	void write_binary_image_clip_to_CSV_file (py::object& pImageFrame);

	py::list read_binary_image_clip_from_CSV_file (py::tuple& pos1,
			py::tuple& pos2);

private:

	py::list convert_image_frame_to_list (py::object& pImageFrame);

	std::deque <std::string> fileBuffer;

	static const std::string __one;
	static const std::string __zero;
	static const std::string __comma;
	static const std::string __nextline;

	std::string itoa (int num);


	void dynamicBindData ();

	int exists(const char *fname);
};


#endif /* IMAGEPROCUTIL_H_ */
