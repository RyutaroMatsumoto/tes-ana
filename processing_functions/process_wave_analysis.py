import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import sys
import logging
import os
import json
import src.trap_filter as trap_filter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
def save_waveform(array: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, array)
    logging.debug("Saved waveform → %s", dest)

def append_metadata(meta_dict: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, indent=2, ensure_ascii=False)
    logging.info("Metadata written → %s (%d channels)", dest, len(meta_dict))

def process_wave(p_id: str, r_id: str, c_ids: list, base_dir: Path, row_index:int=500, show_single_wave=True, show_single_10=False, show_sample_avg=True,trap=False,t_range=[0,50], reprocess=True) -> None:
    """
    Process wave data and generate plots.
    
    Args:
        p_id: Project ID
        r_id: Run ID
        c_ids: List of channel IDs to process
        base_dir: Base directory
        row_index: Index of the row to plot
        show_single_wave: Whether to show single waveform
        show_single_10: Whether to show 10 random waveforms
        show_sample_avg: Whether to show sample averaging waveform
        trap: Whether to apply trap filter
        t_range: Time range in microseconds to display in the plot [start, end]
        reprocess: Whether to reprocess the data
    """
    #input
    raw_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}"
    meta_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata does not exist: {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    dt = metadata['C1--00000']['time_resolution']['dt']
    #output
    plt_dir = base_dir / "generated_data" / "pyplt" /"wave"/ f"p{p_id}" / f"r{r_id}"
    plt_dir.mkdir(parents=True, exist_ok=True)
    par_dir = base_dir / "generated_data" / "pypar" /"wave"/ f"p{p_id}" / f"r{r_id}"
    par_dir.mkdir(parents=True, exist_ok=True)
    # Search for C1, C2, etc. directories and filter by c_ids
    c_dirs = []
    for item in raw_dir.iterdir():
        if item.is_dir() and item.name.startswith('C') and item.name[1:].isdigit():
            c_id = item.name[1:]
            if c_id in c_ids:
                c_dirs.append(item)
    
    if not c_dirs:
        logging.warning(f"No directories matching C{{{','.join(c_ids)}}} found in {raw_dir}")
        return
    
    
    if show_single_wave:# Plot waveforms from all processed channels if show_waveform is True
        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, row_index, sample_avg=False, trap=trap, t_range=t_range)

    if show_single_10:#waveのサンプル数をカウントし、その中から10個選んでそれらをtrap前後でプロット
        show_sample_10(p_id, r_id, c_ids, base_dir, dt,t_range=t_range)
    
    if show_sample_avg:# Plot sample averaging waveform if sample_averaging is True
        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, row_index, sample_avg=True, trap=trap, t_range=t_range)

def plot_waveforms(p_id: str, r_id: str, c_ids: list, base_dir: Path, dt:float, row_index: int, sample_avg=False, trap=False, show_figure=True, rt=1e-8, ft=5e-9, t_range=[0,50]) -> None:

    """
    Plot waveforms from multiple channels on the same plot.
    
    Args:
        p_id: Project ID
        r_id: Run ID
        c_ids: List of channel IDs to plot
        base_dir: Base directory
        row_index: Index of the row to plot (default: 55)
        sample_avg: Bool
        trap: True(ON) or False(OFF)
        rt: Rise time in seconds (default: 1e-8)
        ft: Flat top time in seconds (default: 5e-9)
        t_range: Time range in microseconds to display in the plot [start, end]
    """
    # 保存先ディレクトリの設定
    plt_dir = base_dir / "generated_data" / "pyplt" / "wave" / f"p{p_id}" / f"r{r_id}"
    plt_dir.mkdir(parents=True, exist_ok=True)
    par_dir = base_dir / "generated_data" / "pypar" / "wave" / f"p{p_id}" / f"r{r_id}"
    par_dir.mkdir(parents=True, exist_ok=True)
    #plot
    plt.figure(figsize=(10, 5))
    
    # 色のリスト (必要に応じて拡張) - c1は赤、c2は青に固定
    colors = ['g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  # その他のチャンネル用
    
    # 各チャンネルのデータをプロット
    for i, c_id in enumerate(c_ids):
        # ファイルパスの設定
        file_path = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}" / f"C{c_id}--Trace.npy"
        if not file_path.exists():
            file_path =base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}" / f"C{c_id}--wave.npy"
        if not file_path.exists():
            logging.warning(f"File not found: {file_path}")
            continue
        # データを読み込む
    
        try:
            data = np.load(file_path)
            if sample_avg==False:
                # 指定された行を抽出
                if row_index < data.shape[0]:
                    if not trap:
                        wave_data = data[row_index, :]
                        dataname = f"raw_waveform_p{p_id}_r{r_id}_s{row_index}_C{','.join(c_ids)}"
                    if trap:
                        wave_data = trap_filter(data[row_index, :], dt, rt, ft)
                        dataname = f"trap_waveform_p{p_id}_r{r_id}_s{row_index}_C{','.join(c_ids)}_rt{rt:.1e}_ft{ft:.1e}"
                else:
                    logging.warning(f"Row index {row_index} out of bounds for {file_path} with shape {data.shape}")
                    continue
            if sample_avg==True:
                if not trap:
                    wave_data = np.mean(data, axis=0)  
                    dataname = f'sample_averaging_raw_waveform_p{p_id}_r{r_id}_C{",".join(c_ids)}'
                    logging.info("Saving data")
                    data_file = par_dir / f"mean_wave_C{c_id}.npz"
                    np.savez(data_file,
                        mean_wave = wave_data)
                    logging.info(f"Data saved to {data_file}")
                if trap:
                    wave_data = trap_filter(np.mean(data, axis=0), dt, rt, ft) 
                    dataname = f'sample_averaging_trap_waveform_p{p_id}_r{r_id}_C{",".join(c_ids)}_rt{rt:.1e}_ft{ft:.1e}'
            # 時間データを生成 (秒単位)
            time_data = np.arange(len(wave_data)) * dt
            
            # マイクロ秒単位のt_rangeを秒単位に変換
            t_range_seconds = [t_range[0] * 1e-6, t_range[1] * 1e-6]
            
            # プロット - c1は赤、c2は青に固定
            if c_id == '1':
                color = 'r'  # c1は赤
            elif c_id == '2':
                color = 'b'  # c2は青
            else:
                color = colors[i % len(colors)]  # その他のチャンネルは色のリストを循環使用
            
            plt.plot(time_data, wave_data, marker='', linestyle='-', color=color, label=f'C{c_id} Data')
            
            # 表示範囲を設定
            if max(time_data) > t_range_seconds[1]:
                plt.xlim(t_range_seconds)
            else:
                logging.warning(f"Time range {t_range} μs exceeds available data range (0-{max(time_data)*1e6:.1f} μs) for C{c_id}")
            logging.info(f"Plotted data from C{c_id}")
        except Exception as e:
            logging.error(f"Error plotting data from {file_path}: {e}")

    plt.title(f'{dataname} (t_range: {t_range[0]}-{t_range[1]} μs)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)

    # プロットを保存
    plt.savefig(plt_dir / f"{dataname}(t_range: {t_range[0]}-{t_range[1]} μs)'.png", dpi=300)
    logging.info(f"Saved plot to {plt_dir}/{dataname}(t_range: {t_range[0]}-{t_range[1]} μs)'.png")
    if show_figure:
        plt.show()

def show_sample_10(p_id: str, r_id: str, c_ids: list, base_dir: Path,dt:float,t_range:list)->None:
     # Get sample count from the first channel
        sample_file_path = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_ids[0]}" / f"C{c_ids[0]}--Trace.npy"
        if not sample_file_path.exists():
            sample_file_path = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_ids[0]}" / f"C{c_ids[0]}--wave.npy"
        
        if sample_file_path.exists():
            try:
                data = np.load(sample_file_path)
                total_samples = data.shape[0]
                
                if total_samples >= 10:
                    # Generate 10 random indices within the range of total_samples
                    indices = np.random.choice(total_samples, 10, replace=False)
                    
                    # Plot each of the 10 randomly selected waveforms both raw and trap
                    for i, idx in enumerate(indices):
                        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, idx, sample_avg=False, trap=False, show_figure=False, t_range=t_range)
                        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, idx, sample_avg=False, trap=True, show_figure=False, t_range=t_range)
                        logging.info(f"Plotted random waveform {i+1}/10 (sample index: {idx})")
                else:
                    # If fewer than 10 samples, plot all available
                    for idx in range(total_samples):
                        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, idx, sample_avg=False, trap=False, show_figure=False, t_range=t_range)
                        plot_waveforms(p_id, r_id, c_ids, base_dir, dt, idx, sample_avg=False, trap=True, show_figure=False, t_range=t_range)
                        logging.info(f"Plotted waveform {idx+1}/{total_samples} (sample index: {idx})")
            except Exception as e:
                logging.error(f"Error processing samples for show_single_10: {e}")
        else:
            logging.warning(f"No data file found for channel C{c_ids[0]} to determine sample count")