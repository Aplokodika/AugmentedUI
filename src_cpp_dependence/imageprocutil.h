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

#include <fstream>
#include <deque>
#include <stdint.h>
#include <string>
#include <vector>
#include <ctype.h>

namespace py = boost::python;

struct ImageProcessExtension {

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

	py::list read_binary_image_clip_from_CSV_file (py::object& representationOfValue1,
												   py::object& representationOfValue0);


	void resetReadPos ();

	void setReadPos(py::object pos);

private:

	int readPos;

	int openMode;

	int bindType;

	std::fstream *fHandle;

	std::deque <std::string> fileBuffer;

	void dynamicBindData ();

	static const std::string __one;
	static const std::string __zero;
	static const std::string __comma;
	static const std::string __nextline;
	static const std::string _blankspace;

public:

	static py::list convert_image_frame_to_list (py::object& pImageFrame);


	static std::string itoa (int num);
	static int atoi (std::string val);



	static int exists(const char *fname);

	static std::deque <std::string> get_comma_separated_strings (std::string pLine);

	static bool isnumber (std::string str);

};


#endif /* IMAGEPROCUTIL_H_ */
