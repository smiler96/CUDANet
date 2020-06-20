/*
* utils.h
*
*  Created on: Sep 20, 2019
*      Author: wanqina
*/

#include "utils.h"
#include <string.h>


namespace net_utils {

	void getAllFiles(std::string path, std::vector<std::string>& files)
	{
		// long	hFile; 会报错
		intptr_t hFile = 0;
		//文件信息  
		struct _finddata_t fileinfo;
		std::string p;
		if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				if ((fileinfo.attrib & _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					{
						files.push_back(p.assign(path).append("\\").append(fileinfo.name));
						getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
					}
				}
				else
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}

			} while (_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}

	}

	void getAllFormatFiles(std::string path, std::vector<std::string>& files, std::string format)
	{
		//文件句柄  
		intptr_t hFile = 0;
		//文件信息  
		struct _finddata_t fileinfo;
		std::string p;
		if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
		{
			do
			{
				if ((fileinfo.attrib & _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					{
						//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
						getAllFormatFiles(p.assign(path).append("\\").append(fileinfo.name), files, format);
					}
				}
				else
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			} while(_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}
	}

	void getAllImages(uint8_t* images, const std::vector<std::string> files, const int data_dim)
	{
		cv::Mat image;
		for (int offset = 0; offset < files.size(); offset++)
		{
			image = cv::imread(files[offset], 0);
			if (image.cols*image.rows != data_dim)
			{
				std::cout << "image data_dim error!\n";
				return;
			}
			else
			{
				std::memcpy(images + offset*data_dim, image.data, sizeof(uint8_t)*data_dim);
			}
		}
	}

	void checkArrayNan(const float* d_x, const int n)
	{
		float* m = new float[n];
		cublasGetVector(n, sizeof(*m), d_x, 1, m, 1);

		for (int i = n - 1; i >= 0; --i)
		{
			//checkValidNum(*(m + i));
			if (isnan(*(m + i)) || isinf(*(m + i)))
			{
				std::cout << i << " " << *(m + i) << "\n";
				throw "invalid";
			}
		}

		delete[] m;
	}

} /* namespace global */
