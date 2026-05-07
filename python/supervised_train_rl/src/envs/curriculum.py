import random

class CurriculumMapManager:
    def __init__(self):
        # 難易度順に並べたリスト（目安）
        self.all_maps = [
            'Austin', 'Spielberg', 'MexicoCity', 'Sakhir', 'Sochi',          # Easy
            'Hockenheim', 'Melbourne', 'Silverstone', 'YasMarina', 'Budapest', # Medium
            'MoscowRaceway', 'Nuerburgring', 'Oschersleben', 'Sepang', 'Spa'   # Hard
        ]
        self.total_maps = len(self.all_maps)

    def get_map_by_progress(self, current_step, total_steps):
        """
        学習の進捗度に応じて、選択可能なマップの範囲（ウィンドウ）を広げる
        """
        progress = current_step / total_steps
        
        # 進捗に応じて、初めは5個、最後は15個すべてが対象になるようにウィンドウを広げる
        window_size = int(5 + (self.total_maps - 5) * progress)
        window_size = min(window_size, self.total_maps)
        
        # 選択可能な範囲からランダムに1つ選ぶ
        selectable_maps = self.all_maps[:window_size]
        return random.choice(selectable_maps)