/*
 *VICP Network Connection
 *2017.10.12 Fumiaki Sato
 */


typedef struct
{
	char descriptor_name[16];
	char template_name[16];
	short int comm_type;
	short int comm_order;
	int wave_descriptor;
	int user_text;
	int res_desc;
	int trigtime_array;
	int ris_time_array;
	int res_time_array;
	int wave_array_1;
	int wave_array_2;
	int res_array_2;
	int res_array_3;
	char instrument_name[16];
	int instrument_number;
	char trace_label[16];
	short int reserved1;
	short int reserved2;		
	int wave_array_count;
	int pnts_per_screen;
	int first_valid_pnt;
	int last_valid_pnt;
	int first_point;
	int sparsing_factor;
	int sebment_index;
	int subarray_count;
	int sweeps_per_acq;
	short int points_per_pair;
	short int pair_offset;
	float vertical_gain;
	float vertical_offset;
	float max_value;
	float min_value;
	short int nominal_bits;
	short int nom_subarray_count;
	float horiz_interval;
	double horiz_offset;
	double pixel_offset;
	char verunit[48];
	char horunit[48];
	float horiz_uncertainty;
	double timestamp_sec;
	char timestamp_min;
	char timestamp_hours;
	char timestamp_days;
	char timestamp_months;
	short int timestamp_year;
	float acq_duration;
	short int record_type;
	short int processing_done;
	short int ris_sweeps;
	short int timebase;
	short int vert_coupling;
	float probe_att;
	short int fixed_ver_gain;
	short int bandwidth_limit;
	float vertical_vernier;
	float acq_vert_offset;
	short int wave_source;
	int wf_stpoint;
	int trigtime_stpoint;

} WAVE_DESC;