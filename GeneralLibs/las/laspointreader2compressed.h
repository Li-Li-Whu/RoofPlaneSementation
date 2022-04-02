/*
===============================================================================

  FILE:  laspointreader2compressed.h
  
  CONTENTS:
  
    Reads a point of type 2 (with rgb color) from our compressed LAS format 1.2

  PROGRAMMERS:
  
    martin isenburg@cs.unc.edu
  
  COPYRIGHT:
  
    copyright (C) 2008  martin isenburg@cs.unc.edu
    
    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    7 September 2008 -- created after baking three apple pies with Kaoru 
  
===============================================================================
*/
#ifndef LAS_POINT_READER_2COMPRESSED_H
#define LAS_POINT_READER_2COMPRESSED_H

#include "laspointreader.h"

#include "rangemodel.h"
#include "rangedecoder.h"
#include "integercompressor_newer.h"

#include <stdio.h>

class LASpointReader2compressed : public LASpointReader
{
public:
  bool read_point(LASpoint* point, double* gps_time = 0, unsigned short* rgb = 0);
  LASpointReader2compressed(FILE* file);
  ~LASpointReader2compressed();

private:
  FILE* file;
  LASpoint last_point;
  int last_dir;
  int last_x_diff[2][3];
  int last_y_diff[2][3];
  int last_incr[2];
  unsigned short last_rgb[3];
  void init_decoder();
  RangeDecoder* rd;
  IntegerCompressorContext* ic_dx;
  IntegerCompressorContext* ic_dy;
  IntegerCompressorContext* ic_z;
  RangeModel* rm_changed_values;
  IntegerCompressorContext* ic_intensity;
  RangeModel* rm_bit_byte;
  RangeModel* rm_classification;
  IntegerCompressorContext* ic_scan_angle_rank;
  RangeModel* rm_user_data;
  IntegerCompressorContext* ic_point_source_ID;
  IntegerCompressorContext* ic_r;
  IntegerCompressorContext* ic_g;
  IntegerCompressorContext* ic_b;
};

#endif
