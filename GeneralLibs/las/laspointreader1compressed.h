/*
===============================================================================

  FILE:  laspointreader1compressed.h
  
  CONTENTS:
  
    Reads a point of type 1 (with gps_time) from our compressed LAS format 1.1

  PROGRAMMERS:
  
    martin isenburg@cs.unc.edu
  
  COPYRIGHT:
  
    copyright (C) 2007  martin isenburg@cs.unc.edu
    
    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    7 September 2008 -- updated to support LAS format 1.2 
    23 February 2007 -- created 12 hours into henna's 32nd birthday
  
===============================================================================
*/
#ifndef LAS_POINT_READER_1COMPRESSED_H
#define LAS_POINT_READER_1COMPRESSED_H

#include "laspointreader.h"

#include "rangemodel.h"
#include "rangedecoder.h"
#include "integercompressor_newer.h"

#include <stdio.h>

class LASpointReader1compressed : public LASpointReader
{
public:
  bool read_point(LASpoint* point, double* gps_time = 0, unsigned short* rgb = 0);
  LASpointReader1compressed(FILE* file);
  ~LASpointReader1compressed();

private:
  FILE* file;
  LASpoint last_point;
  int last_dir;
  int last_x_diff[2][3];
  int last_y_diff[2][3];
  int last_incr[2];
  double last_gps_time;
  int last_gps_time_diff;
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
  IntegerCompressorContext* ic_gps_time;
  RangeModel** rm_gps_time_multi;
  int multi_extreme_counter;
};

#endif
