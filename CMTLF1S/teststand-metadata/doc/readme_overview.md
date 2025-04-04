This is an overview of the  LNA `CMTLF1S` measurements. For more details on the individuals runs, see the readme files in this directory. 

The follow the following naming scheme: `readme_<period>_<runs>.md`.  
<style>
@media (prefers-color-scheme: dark) {
  .logo-inline {
    content: url("./../../logo/lbnl_logo_dark.png");
  }
}
</style>

## Overview table: period 1 `p01`: Amp + Oscilloscope
All runs in this period are taken this the **CMTLF1S**. 
|                   |                 |            |            |                                                                                                              |
| :---------------- | :-------------- | :--------- | :--------- | :----------------------------------------------------------------------------------------------------------- |
| **runs**          | **# waveforms** | **Amp** | **cables** | **short description**                                                                                        |
| `r004`            | 500           | yes         | y | Low freq noise data with Amp     |
| `r005`            | 500           | yes         | y | High freq noise data with Amp    |
| `r006`            | 100           | no          | n | High freq noise data without Amp |
| `r007`            | 100           | no          | n | Low freq noise data without Amp  |




## Overview table: period 3 `p03`: ASIC + Vireo
All runs in this period are taken this the **ASIC L1k65n** (same as `p01`). In constrast to `p01` this dataset was recorded using a digitizer **SkuTek FemtoDAQ Vireo** instead of the oscilloscope. 
|          |                 |            |            |                         |
| :------- | :-------------- | :--------- | :--------- | :---------------------- |
| **runs** | **# waveforms** | **buffer** | **cables** | **short description**   |
| `r001`   | 5,000           | yes        | short      | first test              |
| `r002`   | 100,000         | yes        | short      | first higher stat. data |
| `r003`-`r014`  | 3,000         | yes        | short      | Optimization of DAQ Parameters |
| `r015`-`r016`  | 3,000         | yes        | short      | Check for decay time with longer charge-time for Capasitor |
| `r017`-`r026`  | 5,000         | yes        | short      | Optimization of DAQ prameters |
| `r027`-`r030`  | 5,000 or 50,000       | yes        | ~4m      | With longer cables |
| `r031`-`r036`  | 50,000       | yes        | ~4m and 2m     | Comparison of different cable-lengths under same setting  |
| `r037`-`r038` | 3,000 | yes | 2m | reference data for decay time constant under 90.5K |
| `r039`-`r041` | 50,000 | yes | 2m, 4m, 6.4m| Examine influence of cable length on the energy resolution under 90.5K |
| `r042`-`r044` | 100,000 | yes | 2m, 4m, 6.4m| 100,000 samples to reduce statistical errors |
