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
        
        # カーネル設定 (学習環境と一致させる)
        width_sec = 0.05
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt)
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        self.kernel = (target_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)).view(1, 1, -1)
        self.padding = kernel_radius

        # MIDIロードと軌道生成
        self.bpm, self.trajectory_buffer, self.duration_sec = self._load_and_process_midi(midi_path)
        
        print(f"[MidiGen] Loaded: {midi_path}")
        print(f"          BPM: {self.bpm:.1f}, Duration: {self.duration_sec:.1f}s")

    def _load_and_process_midi(self, midi_path):
        mid = mido.MidiFile(midi_path)
        
        # 1. テンポ解析 (最初のテンポ設定を採用)
        tempo = 500000 # Default 120BPM
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        bpm = mido.tempo2bpm(tempo)
        
        # 2. ノートイベント抽出 (絶対時刻へ変換)
        current_time = 0.0
        spikes = [] # (time_sec, velocity)
        
        # トラックのマージ
        for msg in mid.merged_track:
            # 時間を経過させる (delta time -> seconds)
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            current_time += time_delta
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # ドラムマップ (General MIDI)
                # 38: Snare, 31-34: Sticks など。必要に応じてフィルタリング
                # ここではスネア周辺の主要な音を対象とする例
                # if msg.note in [31, 32, 33, 34, 38, 40]: 
                spikes.append(current_time)

        if not spikes:
            raise ValueError("No note_on events found in MIDI file.")

        total_duration = current_time + 2.0 # 余白
        total_steps = int(total_duration / self.dt)
        
        # 3. スパイク行列の作成 (Torch)
        spike_tensor = torch.zeros((1, 1, total_steps), device=self.device)
        
        for t in spikes:
            idx = int(t / self.dt)
            if idx < total_steps:
                spike_tensor[0, 0, idx] = 1.0 # Velocity対応するなら msg.velocity / 127.0

        # 4. 畳み込みによる軌道生成 (GPU/CPU)
        with torch.no_grad():
            traj = F.conv1d(spike_tensor, self.kernel, padding=self.padding)
            traj = traj.view(-1) # [Length]
            
        return bpm, traj, total_duration

    def get_state(self, t_now):
        """
        現在時刻 t_now における (Phase, Lookahead_Trajectory) を取得
        """
        # A. 位相計算 (Phase)
        # 1拍ごとの位相 (Sim環境の定義に合わせる: phase = t * bpm_scale * 2pi)
        # porcaro_rl_env.py: phase = time_s * (bpm / 60.0) * (2 * math.pi)
        phase_rad = t_now * (self.bpm / 60.0) * (2 * np.pi)
        
        # B. 先読み軌道の取得
        current_step = int(t_now / self.dt)
        
        # バッファ範囲外処理
        if current_step >= len(self.trajectory_buffer):
            # 終了後はゼロ埋め
            return phase_rad, torch.zeros(self.lookahead_steps, device=self.device)

        end_step = current_step + self.lookahead_steps
        
        if end_step < len(self.trajectory_buffer):
            chunk = self.trajectory_buffer[current_step : end_step]
        else:
            # 配列末尾を超える場合はパディング
            valid_len = len(self.trajectory_buffer) - current_step
            chunk = torch.zeros(self.lookahead_steps, device=self.device)
            chunk[:valid_len] = self.trajectory_buffer[current_step:]
            
        # 正規化 (Target Forceで割る -> 0.0~1.0)
        # 学習環境では rhythm_buf = rhythm_buf / target_hit_force しているので
        # ここでも正規化して返す
        return phase_rad, chunk / self.target_force