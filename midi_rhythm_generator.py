# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/midi_rhythm_generator.py
# (※または配置されているディレクトリの同名ファイル)

import torch
import torch.nn.functional as F
import numpy as np
import mido

class MidiRhythmGenerator:
    """
    MIDIファイルを読み込み、強化学習エージェント用のターゲット軌道を生成するクラス。
    学習環境(rhythm_generator.py)と完全に等価な信号処理を行います。
    """
    def __init__(self, midi_path, device, dt=0.02, target_force=20.0, lookahead_steps=25):
        self.device = device
        self.dt = dt
        self.target_force = target_force
        self.lookahead_steps = lookahead_steps
        
        # --- 変更箇所: 学習環境と完全に一致させる ---
        width_sec = 0.035 # 0.05 -> 0.035
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt)
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        # 指数を2乗から4乗(Super-Gaussian)へ変更
        self.kernel = (target_force * torch.exp(-0.5 * (t_vals / sigma) ** 4)).view(1, 1, -1)
        # ------------------------------------------------
        
        self.padding = kernel_radius

        # MIDIロードと軌道生成
        self.bpm, self.trajectory_buffer, self.duration_sec = self._load_and_process_midi(midi_path)
        
        print(f"[MidiGen] Loaded: {midi_path}")
        print(f"          BPM: {self.bpm:.1f}, Duration: {self.duration_sec:.1f}s")

    def _load_and_process_midi(self, midi_path):
        mid = mido.MidiFile(midi_path)
        
        tempo = 500000 
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        bpm = mido.tempo2bpm(tempo)
        
        current_time = 0.0
        spikes = [] 
        
        for msg in mid.merged_track:
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            current_time += time_delta
            
            if msg.type == 'note_on' and msg.velocity > 0:
                spikes.append(current_time)

        if not spikes:
            raise ValueError("No note_on events found in MIDI file.")

        total_duration = current_time + 2.0 
        total_steps = int(total_duration / self.dt)
        
        spike_tensor = torch.zeros((1, 1, total_steps), device=self.device)
        
        for t in spikes:
            idx = int(t / self.dt)
            if idx < total_steps:
                spike_tensor[0, 0, idx] = 1.0 

        with torch.no_grad():
            traj = F.conv1d(spike_tensor, self.kernel, padding=self.padding)
            traj = traj.view(-1) 
            
        return bpm, traj, total_duration

    def get_state(self, t_now):
        phase_rad = t_now * (self.bpm / 60.0) * (2 * np.pi)
        
        current_step = int(t_now / self.dt)
        
        if current_step >= len(self.trajectory_buffer):
            return phase_rad, torch.zeros(self.lookahead_steps, device=self.device)

        end_step = current_step + self.lookahead_steps
        
        if end_step < len(self.trajectory_buffer):
            chunk = self.trajectory_buffer[current_step : end_step]
        else:
            valid_len = len(self.trajectory_buffer) - current_step
            chunk = torch.zeros(self.lookahead_steps, device=self.device)
            chunk[:valid_len] = self.trajectory_buffer[current_step:]
            
        return phase_rad, chunk / self.target_force