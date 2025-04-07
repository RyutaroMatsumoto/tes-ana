/*
 *Convert Lecroy Waveform to CSV
 
 *2017.10.12 Fumiaki Sato
 *Waveform length is up to 1MS
 */

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "ConvTrcToCSV.h"


short bintoshort(char *data,int cmdr)
{
  short tmp=0;
  char *chtmp;
  chtmp=(char *)&tmp;
  if(cmdr==0){
    chtmp[0]=data[1];
    chtmp[1]=data[0];
  }else{
    chtmp[0]=data[0];
    chtmp[1]=data[1];
  }
  return tmp;
}

long bintolong(char *data,int cmdr)
{
  long tmp=0;
  char *chtmp;
  chtmp=(char *)&tmp;
  if(cmdr==0){
    chtmp[0]=data[3];
    chtmp[1]=data[2];
    chtmp[2]=data[1];
    chtmp[3]=data[0];
  }else{
    chtmp[0]=data[0];
    chtmp[1]=data[1];
    chtmp[2]=data[2];
    chtmp[3]=data[3];
  }
  return tmp;
}

float bintofloat(char *data,int cmdr)
{
  float tmp=0;
  char *chtmp;
  chtmp=(char *)&tmp;
  if(cmdr==0){
    chtmp[0]=data[3];
    chtmp[1]=data[2];
    chtmp[2]=data[1];
    chtmp[3]=data[0];
  }else{
    chtmp[0]=data[0];
    chtmp[1]=data[1];
    chtmp[2]=data[2];
    chtmp[3]=data[3];
  }
  return tmp;
}

double bintodouble(char *data,int cmdr)
{
  double tmp=0;
  char *chtmp;
  chtmp=(char *)&tmp;
  if(cmdr==0){
    chtmp[0]=data[7];
    chtmp[1]=data[6];
    chtmp[2]=data[5];
    chtmp[3]=data[4];
    chtmp[4]=data[3];
    chtmp[5]=data[2];
    chtmp[6]=data[1];
    chtmp[7]=data[0];
  }else{
    chtmp[0]=data[0];
    chtmp[1]=data[1];
    chtmp[2]=data[2];
    chtmp[3]=data[3];
    chtmp[4]=data[4];
    chtmp[5]=data[5];
    chtmp[6]=data[6];
    chtmp[7]=data[7];
  }
  return tmp;
}

int GetDescriptor(char *wfbin,WAVE_DESC *wd)
{
  int stpoint;
  int i;
  char buf[16];
  for(i=0;i<40;i++){
    if (wfbin[i]==0x23 && wfbin[i+1]==0x39)
      stpoint=i+11;
  }
  wd->comm_order=bintoshort(wfbin+stpoint+34,1);
  memcpy(wd->descriptor_name,wfbin+stpoint,16);
  memcpy(wd->template_name,wfbin+stpoint+16,16);
  wd->comm_type=bintoshort(wfbin+stpoint+32,wd->comm_order);
  wd->wave_descriptor=bintolong(wfbin+stpoint+36,wd->comm_order);
  wd->user_text=bintolong(wfbin+stpoint+40,wd->comm_order);
  wd->res_desc=bintolong(wfbin+stpoint+44,wd->comm_order);
  wd->trigtime_array=bintolong(wfbin+stpoint+48,wd->comm_order);
  wd->ris_time_array=bintolong(wfbin+stpoint+52,wd->comm_order);
  wd->res_time_array=bintolong(wfbin+stpoint+56,wd->comm_order);
  wd->wave_array_1=bintolong(wfbin+stpoint+60,wd->comm_order);
  wd->wave_array_2=bintolong(wfbin+stpoint+64,wd->comm_order);
  wd->res_array_2=bintolong(wfbin+stpoint+68,wd->comm_order);
  wd->res_array_3=bintolong(wfbin+stpoint+72,wd->comm_order);
  memcpy(wd->instrument_name,wfbin+stpoint+76,16);
  wd->instrument_number=bintolong(wfbin+stpoint+92,wd->comm_order);
  memcpy(wd->trace_label,wfbin+stpoint+96,16);
  wd->reserved1=bintoshort(wfbin+stpoint+112,wd->comm_order);
  wd->reserved2=bintoshort(wfbin+stpoint+114,wd->comm_order);		
  wd->wave_array_count=bintolong(wfbin+stpoint+116,wd->comm_order);
  wd->pnts_per_screen=bintolong(wfbin+stpoint+120,wd->comm_order);
  wd->first_valid_pnt=bintolong(wfbin+stpoint+124,wd->comm_order);
  wd->last_valid_pnt=bintolong(wfbin+stpoint+128,wd->comm_order);
  wd->first_point=bintolong(wfbin+stpoint+132,wd->comm_order);
  wd->sparsing_factor=bintolong(wfbin+stpoint+136,wd->comm_order);
  wd->sebment_index=bintolong(wfbin+stpoint+140,wd->comm_order);
  wd->subarray_count=bintolong(wfbin+stpoint+144,wd->comm_order);
  wd->sweeps_per_acq=bintolong(wfbin+stpoint+148,wd->comm_order);
  wd->points_per_pair=bintoshort(wfbin+stpoint+152,wd->comm_order);
  wd->pair_offset=bintoshort(wfbin+stpoint+154,wd->comm_order);
  wd->vertical_gain=bintofloat(wfbin+stpoint+156,wd->comm_order);
  wd->vertical_offset=bintofloat(wfbin+stpoint+160,wd->comm_order);
  wd->max_value=bintofloat(wfbin+stpoint+164,wd->comm_order);
  wd->min_value=bintofloat(wfbin+stpoint+168,wd->comm_order);
  wd->nominal_bits=bintoshort(wfbin+stpoint+172,wd->comm_order);
  wd->nom_subarray_count=bintoshort(wfbin+stpoint+174,wd->comm_order);
  wd->horiz_interval=bintofloat(wfbin+stpoint+176,wd->comm_order);
  wd->horiz_offset=bintodouble(wfbin+stpoint+180,wd->comm_order);
  wd->pixel_offset=bintodouble(wfbin+stpoint+188,wd->comm_order);
  wd->horiz_uncertainty=bintofloat(wfbin+stpoint+292,wd->comm_order);
  wd->timestamp_sec=bintodouble(wfbin+stpoint+296,wd->comm_order);
  wd->timestamp_min=*(wfbin+stpoint+304);
  wd->timestamp_hours=*(wfbin+stpoint+305);
  wd->timestamp_days=*(wfbin+stpoint+306);
  wd->timestamp_months=*(wfbin+stpoint+307);
  wd->timestamp_year=bintoshort(wfbin+stpoint+308,wd->comm_order);
  wd->acq_duration=bintofloat(wfbin+stpoint+312,wd->comm_order);
  wd->record_type=bintoshort(wfbin+stpoint+316,wd->comm_order);
  wd->processing_done=bintoshort(wfbin+stpoint+318,wd->comm_order);
  wd->ris_sweeps=bintoshort(wfbin+stpoint+322,wd->comm_order);
  wd->timebase=bintoshort(wfbin+stpoint+324,wd->comm_order);
  wd->vert_coupling=bintoshort(wfbin+stpoint+326,wd->comm_order);
  wd->probe_att=bintofloat(wfbin+stpoint+328,wd->comm_order);
  wd->fixed_ver_gain=bintoshort(wfbin+stpoint+332,wd->comm_order);
  wd->bandwidth_limit=bintoshort(wfbin+stpoint+334,wd->comm_order);
  wd->vertical_vernier=bintofloat(wfbin+stpoint+336,wd->comm_order);
  wd->acq_vert_offset=bintofloat(wfbin+stpoint+340,wd->comm_order);
  wd->wave_source=bintoshort(wfbin+stpoint+344,wd->comm_order);
  wd->wf_stpoint=stpoint+wd->wave_descriptor+wd->trigtime_array+wd->ris_time_array;
  wd->trigtime_stpoint=stpoint+wd->wave_descriptor;
  
  return 0;
}

void ShowDescriptor(WAVE_DESC *wd)
{
  printf("descriptor_name:%s\n",wd->descriptor_name);
  printf("template_name:%s\n",wd->template_name);
  printf("comm_type:%d\n",wd->comm_type);
  printf("comm_order:%d\n",wd->comm_order);
  printf("wave_descriptor:%d\n",wd->wave_descriptor);
  printf("user_text:%d\n",wd->user_text);
  printf("res_desc:%d\n",wd->res_desc);
  printf("trigtime_array:%d\n",wd->trigtime_array);
  printf("ris_time_array:%d\n",wd->ris_time_array);
  printf("res_time_array:%d\n",wd->res_time_array);
  printf("wave_array_1:%d\n",wd->wave_array_1);
  printf("wave_array_2:%d\n",wd->wave_array_2);
  printf("res_array_2:%d\n",wd->res_array_2);
  printf("res_array_3:%d\n",wd->res_array_3);
  printf("instrument_name:%s\n",wd->instrument_name);
  printf("instrument_number:%d\n",wd->instrument_number);
  printf("trace_label:%s\n",wd->trace_label);
  printf("wave_array_count:%d\n",wd->wave_array_count);
  printf("pnts_per_screen:%d\n",wd->pnts_per_screen);
  printf("first_valid_pnt:%d\n",wd->first_valid_pnt);
  printf("last_valid_pnt:%d\n",wd->last_valid_pnt);
  printf("first_point:%d\n",wd->first_point);
  printf("sparsing_factor:%d\n",wd->sparsing_factor);
  printf("sebment_index:%d\n",wd->sebment_index);
  printf("subarray_count:%d\n",wd->subarray_count);
  printf("sweeps_per_acq:%d\n",wd->sweeps_per_acq);
  printf("points_per_pair:%d\n",wd->points_per_pair);
  printf("vertical_gain:%e\n",wd->vertical_gain);
  printf("vertical_offset:%e\n",wd->vertical_offset);
  printf("max_value:%e\n",wd->max_value);
  printf("min_value:%e\n",wd->min_value);
  printf("nominal_bits:%d\n",wd->nominal_bits);
  printf("nom_subarray_count:%d\n",wd->nom_subarray_count);
  printf("horiz_interval:%e\n",wd->horiz_interval);
  printf("horiz_offset:%e\n",wd->horiz_offset);
  printf("pixel_offset:%e\n",wd->pixel_offset);
  printf("timestamp_sec:%e\n",wd->timestamp_sec);
  printf("timestamp_min:%d\n",wd->timestamp_min);
  printf("timestamp_hours:%d\n",wd->timestamp_hours);
  printf("timestamp_days:%d\n",wd->timestamp_days);
  printf("timestamp_months:%d\n",wd->timestamp_months);
  printf("timestamp_year:%d\n",wd->timestamp_year);
  printf("acq_duration:%e\n",wd->acq_duration);
  printf("record_type:%d\n",wd->record_type);
  printf("processing_done:%d\n",wd->processing_done);
  printf("ris_sweeps:%d\n",wd->ris_sweeps);
  printf("timebase:%d\n",wd->timebase);
  printf("vert_coupling:%d\n",wd->vert_coupling);
  printf("probe_att:%e\n",wd->probe_att);
  printf("fixed_ver_gain:%d\n",wd->fixed_ver_gain);
  printf("bandwidth_limit:%d\n",wd->bandwidth_limit);
  printf("vertical_vernier:%e\n",wd->vertical_vernier);
  printf("acq_vert_offset:%e\n",wd->acq_vert_offset);
  printf("wave_source:%d\n",wd->wave_source);
}

int main(){
 
  // オシロデータのチャネル数
  //int num_channel = 2;

  // 波形数
  int N = 1000;

  for (int ch = 4; ch <= 4; ch++) {
    for (int num = 0; num < N; num++) {

      FILE* fp_r;
      FILE* fp_w;
      char binWaveform[2001000];
      WAVE_DESC wd;
      
      printf("file num = %d\n", num);
      
      // File read
      char fname_read[2048];
      sprintf(fname_read, "./trc/C%d--wave--%05d.trc", ch, num);
      
      fp_r=fopen(fname_read,"rb");
      if(fp_r==NULL){
	printf("Can't open the file to read \n");
	return -1;	
      }
      fread(binWaveform,1,2001000,fp_r);
      GetDescriptor(binWaveform,&wd);
      //ShowDescriptor(&wd);
      fclose(fp_r);
      
      //Save waveform data with time
      char fname_save[2048];
      sprintf(fname_save, "./csv/C%d--wave--%05d.csv", ch, num);
      
      fp_w=fopen(fname_save,"w");
      if(fp_w==NULL){
	printf("Can't open the file to save\n");
	return -1;
      }

      char fileBuf[512];        
      int i,seg,wf_length;
      double trigtime;
      wf_length=wd.wave_array_count/wd.subarray_count;
      for(seg=0;seg < wd.subarray_count;seg++){
	if(wd.subarray_count>1){
	  trigtime=bintodouble(binWaveform+wd.trigtime_stpoint+seg*16+8,wd.comm_order);
	}else{
	  trigtime=wd.horiz_offset;
	}
	if(wd.comm_type==0){
	  signed char tmp;
	  for(i=0;i<wf_length;i++){
	    tmp=binWaveform[wd.wf_stpoint+wf_length*seg+i];
	    sprintf(fileBuf,"%e, %e\n",i*wd.horiz_interval+trigtime,tmp*wd.vertical_gain-wd.vertical_offset);
	    fputs(fileBuf,fp_w);
	  }
	  
	}else{
	  short tmp; 
	  for(i=0;i<wf_length;i++){
	    tmp=bintoshort(binWaveform+wd.wf_stpoint+wf_length*2*seg+i*2,wd.comm_order);
	    sprintf(fileBuf,"%e, %e\n",i*wd.horiz_interval+trigtime,tmp*wd.vertical_gain-wd.vertical_offset);
	    fputs(fileBuf,fp_w);
	  }
	}
      }

      fclose(fp_w);
      //free(binWaveform);
    }
  }
  
}
