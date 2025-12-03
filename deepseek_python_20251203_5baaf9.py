import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Заглавие на приложението
st.set_page_config(page_title="Ski Glide + Slope + CS Zones", layout="wide")
st.title("🎿 onFlows -- Комбиниран модел")
st.subheader("Ski Glide + Slope + CS Zones")

# ============================================================================
# КЛАСОВЕ ЗА ПРЕДВАРИТЕЛНА ОБРАБОТКА И МОДЕЛИ
# ============================================================================

class TCXParser:
    """Парсер на TCX файлове"""
    
    @staticmethod
    def parse_tcx(file):
        """Парсва TCX файл и връща DataFrame с точките"""
        tree = ET.parse(file)
        root = tree.getroot()
        
        # Намираме namespace
        ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
        
        points = []
        for trackpoint in root.findall('.//ns:Trackpoint', ns):
            time_elem = trackpoint.find('ns:Time', ns)
            if time_elem is None:
                continue
                
            time_val = pd.to_datetime(time_elem.text)
            
            # Позиция
            pos_elem = trackpoint.find('ns:Position', ns)
            if pos_elem is None:
                continue
                
            lat_elem = pos_elem.find('ns:LatitudeDegrees', ns)
            lon_elem = pos_elem.find('ns:LongitudeDegrees', ns)
            if lat_elem is None or lon_elem is None:
                continue
                
            latitude = float(lat_elem.text)
            longitude = float(lon_elem.text)
            
            # Височина
            alt_elem = trackpoint.find('ns:AltitudeMeters', ns)
            altitude = float(alt_elem.text) if alt_elem is not None else 0.0
            
            # Дистанция
            dist_elem = trackpoint.find('ns:DistanceMeters', ns)
            distance = float(dist_elem.text) if dist_elem is not None else 0.0
            
            # ЧСС
            hr_elem = trackpoint.find('.//ns:HeartRateBpm/ns:Value', ns)
            heart_rate = int(hr_elem.text) if hr_elem is not None else None
            
            points.append({
                'time': time_val,
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'distance': distance,
                'heart_rate': heart_rate
            })
        
        df = pd.DataFrame(points)
        
        if len(df) > 0:
            # Изчисляваме времеви разлики
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            df['time_from_start'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
            
            # Изчисляваме хоризонтално разстояние между точки
            df['lat_rad'] = np.radians(df['latitude'])
            df['lon_rad'] = np.radians(df['longitude'])
            
            # Изчисляваме разстояние по Haversine формула
            lat_diff = df['lat_rad'].diff()
            lon_diff = df['lon_rad'].diff()
            
            a = np.sin(lat_diff/2)**2 + np.cos(df['lat_rad'].shift()) * np.cos(df['lat_rad']) * np.sin(lon_diff/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            earth_radius = 6371000  # метра
            df['horizontal_dist'] = c * earth_radius
            df.loc[0, 'horizontal_dist'] = 0
            
            # Накопена хоризонтална дистанция
            df['cumulative_dist'] = df['horizontal_dist'].cumsum()
            
            # Моментна скорост
            df['instant_speed'] = df['horizontal_dist'] / df['time_diff']
            df.loc[df['time_diff'] == 0, 'instant_speed'] = 0
            
        return df

class DataPreprocessor:
    """Клас за предварителна обработка на данните"""
    
    def __init__(self, df, params=None):
        self.df = df.copy()
        self.params = params or {}
        
        # Параметри по подразбиране
        self.default_params = {
            'h_min': 0.1,  # минимална промяна във височината
            'g_max': 100,  # максимален наклон (%)
            'v_max': 50,   # максимална скорост (m/s)
            'median_window': 3
        }
        
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def preprocess(self):
        """Извършва предварителна обработка на данните"""
        if len(self.df) == 0:
            return self.df
        
        # 1. Сортиране по време
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # 2. Изглаждане на височината с медианен филтър
        window = self.params['median_window']
        if window % 2 == 1 and len(self.df) >= window:
            self.df['altitude_smooth'] = self.df['altitude'].rolling(
                window=window, center=True, min_periods=1
            ).median()
        else:
            self.df['altitude_smooth'] = self.df['altitude']
        
        # 3. Пресмятане на наклона между точки
        self.df['alt_diff'] = self.df['altitude_smooth'].diff()
        self.df['slope_pct'] = (self.df['alt_diff'] / self.df['horizontal_dist']) * 100
        self.df.loc[self.df['horizontal_dist'] == 0, 'slope_pct'] = 0
        
        # 4. Филтриране на невалидни интервали
        valid_mask = (
            (self.df['time_diff'] > 0) &
            (self.df['horizontal_dist'] > 0) &
            (self.df['instant_speed'].abs() <= self.params['v_max']) &
            (self.df['alt_diff'].abs() >= self.params['h_min']) &
            (self.df['slope_pct'].abs() <= self.params['g_max'])
        )
        
        # Първият ред винаги е валиден
        valid_mask.iloc[0] = True
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        return self.df

class Segmenter:
    """Клас за сегментиране на активността"""
    
    def __init__(self, df, segment_duration=5):
        self.df = df.copy()
        self.segment_duration = segment_duration
        
        # Параметри за валидни сегменти
        self.min_points = 5
        self.min_distance = 5  # метра
        self.min_time = 3  # секунди
        self.max_slope = 30  # %
    
    def create_segments(self):
        """Създава сегменти с фиксирана продължителност"""
        if len(self.df) == 0:
            return pd.DataFrame()
        
        # Номер на сегмента за всяка точка
        self.df['seg_id'] = (self.df['time_from_start'] // self.segment_duration).astype(int)
        
        # Групиране по сегменти
        segments = []
        
        for seg_id, group in self.df.groupby('seg_id'):
            if len(group) < self.min_points:
                continue
            
            # Пресмятане на характеристики на сегмента
            seg_data = {
                'seg_id': seg_id,
                't_start': group['time'].iloc[0],
                't_end': group['time'].iloc[-1],
                'duration': (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds(),
                'distance': group['horizontal_dist'].sum(),
                'altitude_start': group['altitude_smooth'].iloc[0],
                'altitude_end': group['altitude_smooth'].iloc[-1],
                'altitude_diff': group['altitude_smooth'].iloc[-1] - group['altitude_smooth'].iloc[0],
                'n_points': len(group),
                'instant_speeds': group['instant_speed'].values
            }
            
            # Средна скорост
            if seg_data['duration'] > 0:
                seg_data['avg_speed'] = seg_data['distance'] / seg_data['duration']
            else:
                seg_data['avg_speed'] = 0
            
            # Среден наклон (%)
            if seg_data['distance'] > 0:
                seg_data['slope_pct'] = (seg_data['altitude_diff'] / seg_data['distance']) * 100
            else:
                seg_data['slope_pct'] = 0
            
            # Дисперсия на скоростта
            if len(group) > 1:
                seg_data['speed_variance'] = np.var(group['instant_speed'])
            else:
                seg_data['speed_variance'] = 0
            
            segments.append(seg_data)
        
        segments_df = pd.DataFrame(segments)
        
        if len(segments_df) > 0:
            # Филтриране на валидни сегменти
            valid_mask = (
                (segments_df['n_points'] >= self.min_points) &
                (segments_df['distance'] >= self.min_distance) &
                (segments_df['duration'] >= self.min_time) &
                (segments_df['slope_pct'].abs() <= self.max_slope)
            )
            
            segments_df = segments_df[valid_mask].reset_index(drop=True)
            
            # Устойчивост на сегменти
            if len(segments_df) > 0:
                segments_df['stability'] = 1 / (1 + segments_df['speed_variance'])
        
        return segments_df

class GlideModel:
    """Модел 1 -- Плъзгаемост (Ski Glide Dynamics)"""
    
    def __init__(self, segments_df, alpha_glide=0.5):
        self.segments_df = segments_df.copy()
        self.alpha_glide = alpha_glide
        self.downhill_slope_range = (-15, -5)  # %
        
    def run(self):
        """Изпълнява модела за плъзгаемост"""
        if len(self.segments_df) == 0:
            return self.segments_df, {}
        
        # 1. Избор на downhill сегменти
        mask_downhill = (
            (self.segments_df['slope_pct'] >= self.downhill_slope_range[0]) &
            (self.segments_df['slope_pct'] <= self.downhill_slope_range[1])
        )
        
        downhill_segments = self.segments_df[mask_downhill].copy()
        
        # Допълнително условие за инерция
        valid_downhill = []
        for i in range(1, len(downhill_segments)):
            if downhill_segments.iloc[i-1]['seg_id'] == downhill_segments.iloc[i]['seg_id'] - 1:
                valid_downhill.append(i-1)
                valid_downhill.append(i)
        
        downhill_segments = downhill_segments.iloc[list(set(valid_downhill))].copy()
        
        if len(downhill_segments) < 3:
            # Недостатъчно данни за модела
            self.segments_df['V_glide'] = self.segments_df['avg_speed']
            return self.segments_df, {}
        
        # 2. Премахване на аутлайъри
        downhill_segments['ratio'] = downhill_segments['avg_speed'] / downhill_segments['slope_pct'].abs()
        
        q5 = downhill_segments['ratio'].quantile(0.05)
        q95 = downhill_segments['ratio'].quantile(0.95)
        
        mask_outliers = (downhill_segments['ratio'] >= q5) & (downhill_segments['ratio'] <= q95)
        downhill_clean = downhill_segments[mask_outliers].copy()
        
        # 3. Линеен Glide модел
        if len(downhill_clean) >= 2:
            slope_vals = downhill_clean['slope_pct'].values
            speed_vals = downhill_clean['avg_speed'].values
            
            # Линейна регресия
            slope = slope_vals.reshape(-1, 1)
            reg = np.linalg.lstsq(np.hstack([slope, np.ones_like(slope)]), speed_vals, rcond=None)[0]
            a, b = reg[0], reg[1]
            
            # Проверка за статистическа значимост
            # Тук опростяваме - в реален сценарий ще използваме statsmodels или scipy
            if len(downhill_clean) >= 10:  # минимален брой точки за стабилна регресия
                r_squared = np.corrcoef(slope_vals, speed_vals)[0, 1]**2
                if r_squared < 0.3:
                    # Слаба корелация - използваме нулева корекция
                    a, b = 0, np.mean(speed_vals)
        else:
            a, b = 0, np.mean(downhill_segments['avg_speed'])
        
        # 4. Индекс на плъзгаемост по активност
        # Тъй като имаме една активност, изчисляваме за цялата
        if len(downhill_clean) > 0:
            avg_slope = np.average(downhill_clean['slope_pct'], 
                                  weights=downhill_clean['duration'])
            avg_speed_real = np.average(downhill_clean['avg_speed'],
                                       weights=downhill_clean['duration'])
            model_speed = a * avg_slope + b
            
            if model_speed != 0:
                K_raw = avg_speed_real / model_speed
            else:
                K_raw = 1.0
                
            # Омекотен индекс
            K_soft = 1 + self.alpha_glide * (K_raw - 1)
            
            # 5. Корекция на скоростта
            self.segments_df['V_glide'] = self.segments_df['avg_speed'] / K_soft
            
            results = {
                'a': a,
                'b': b,
                'K_raw': K_raw,
                'K_soft': K_soft,
                'n_downhill': len(downhill_clean),
                'avg_slope_downhill': avg_slope,
                'avg_speed_real': avg_speed_real,
                'model_speed': model_speed
            }
        else:
            self.segments_df['V_glide'] = self.segments_df['avg_speed']
            results = {}
        
        return self.segments_df, results

class SlopeModel:
    """Модел 2 -- Влияние на наклона (ΔV%)"""
    
    def __init__(self, segments_df):
        self.segments_df = segments_df.copy()
        self.flat_slope_threshold = 1.0  # %
        self.slope_range_training = (-3, 10)  # %
        
    def run(self):
        """Изпълнява модела за влияние на наклона"""
        if len(self.segments_df) == 0:
            return self.segments_df, {}
        
        # 1. Референтна скорост на равно
        mask_flat = (self.segments_df['slope_pct'].abs() <= self.flat_slope_threshold)
        flat_segments = self.segments_df[mask_flat]
        
        if len(flat_segments) > 0:
            V_flat = np.average(flat_segments['V_glide'], 
                               weights=flat_segments['duration'])
        else:
            # Алтернатива: средно на цялата активност
            V_flat = np.average(self.segments_df['V_glide'], 
                               weights=self.segments_df['duration'])
        
        # 2. Сегменти за обучение на ΔV% модела
        mask_training = (
            (self.segments_df['slope_pct'] > self.slope_range_training[0]) &
            (self.segments_df['slope_pct'] < self.slope_range_training[1]) &
            (self.segments_df['slope_pct'].abs() > self.flat_slope_threshold)
        )
        
        training_segments = self.segments_df[mask_training].copy()
        
        if len(training_segments) >= 5:
            # 3. Реално отклонение на скоростта
            training_segments['delta_V_percent'] = (
                (training_segments['V_glide'] - V_flat) / V_flat * 100
            )
            
            # 4. Квадратичен модел
            X = training_segments['slope_pct'].values
            y = training_segments['delta_V_percent'].values
            
            # Полиномна регресия от втора степен
            coeffs = np.polyfit(X, y, 2)
            c2, c1, c0 = coeffs
            
            # R-squared
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 5. Финална корекция по наклон
            def slope_correction(slope):
                return 1 + np.polyval(coeffs, slope) / 100
            
            self.segments_df['V_final'] = self.segments_df['V_glide'] / \
                self.segments_df['slope_pct'].apply(slope_correction)
            
            results = {
                'V_flat': V_flat,
                'c0': c0,
                'c1': c1,
                'c2': c2,
                'r_squared': r_squared,
                'n_training': len(training_segments)
            }
        else:
            # Недостатъчно данни - няма корекция
            self.segments_df['V_final'] = self.segments_df['V_glide']
            results = {}
        
        return self.segments_df, results

class CSZonesModel:
    """Модел 3 -- Физиологично зониране по критична скорост (CS Zones)"""
    
    def __init__(self, segments_df, cs_speed_kmh=15.0):
        self.segments_df = segments_df.copy()
        self.cs_speed = cs_speed_kmh / 3.6  # преобразуване в m/s
        
        # Дефиниране на зоните (по отношение на CS)
        self.zone_bounds = {
            'Z1': (0.0, 0.8),    # Възстановяване
            'Z2': (0.8, 0.9),    # Лека интензивност
            'Z3': (0.9, 1.0),    # Умерена интензивност
            'Z4': (1.0, 1.05),   # Порогова
            'Z5': (1.05, 1.15),  # Над порогова
            'Z6': (1.15, float('inf'))  # Максимална
        }
        
        # Цветове за визуализация
        self.zone_colors = {
            'Z1': '#2E86AB',   # Синьо
            'Z2': '#A23B72',   # Лилаво
            'Z3': '#F18F01',   # Оранжево
            'Z4': '#C73E1D',   # Червено-оранжево
            'Z5': '#9A031E',   # Червено
            'Z6': '#5D001E'    # Тъмно червено
        }
    
    def run(self):
        """Изпълнява модела за CS зони"""
        if len(self.segments_df) == 0:
            return pd.DataFrame(), {}
        
        # 1. Ефективна скорост за зониране
        self.segments_df['V_eff'] = self.segments_df['V_final']
        
        # Корекция за силни спускания
        mask_downhill = (self.segments_df['slope_pct'] < -5)
        self.segments_df.loc[mask_downhill, 'V_eff'] = np.minimum(
            self.segments_df.loc[mask_downhill, 'V_eff'],
            self.cs_speed * self.zone_bounds['Z1'][1]
        )
        
        # 2. Относителна интензивност
        self.segments_df['intensity_ratio'] = self.segments_df['V_eff'] / self.cs_speed
        
        # 3. Определяне на зона за всеки сегмент
        def get_zone(ratio):
            for zone, (lower, upper) in self.zone_bounds.items():
                if lower <= ratio < upper:
                    return zone
            return 'Z6'  # по подразбиране
        
        self.segments_df['zone'] = self.segments_df['intensity_ratio'].apply(get_zone)
        
        # 4. Изчисляване на статистики по зони
        zone_stats = []
        
        for zone in self.zone_bounds.keys():
            zone_data = self.segments_df[self.segments_df['zone'] == zone]
            
            if len(zone_data) > 0:
                total_time = zone_data['duration'].sum()
                total_time_percent = (total_time / self.segments_df['duration'].sum()) * 100
                avg_speed = np.average(zone_data['V_eff'], weights=zone_data['duration'])
                
                zone_stats.append({
                    'Zone': zone,
                    'Total Time (s)': total_time,
                    'Total Time (min)': total_time / 60,
                    'Percentage (%)': total_time_percent,
                    'Avg Speed (m/s)': avg_speed,
                    'Avg Speed (km/h)': avg_speed * 3.6,
                    'Segments Count': len(zone_data)
                })
        
        zone_stats_df = pd.DataFrame(zone_stats)
        
        # 5. Изходни резултати
        results = {
            'cs_speed_mps': self.cs_speed,
            'cs_speed_kmh': self.cs_speed * 3.6,
            'zone_stats': zone_stats_df,
            'zone_colors': self.zone_colors
        }
        
        return self.segments_df, results

# ============================================================================
# STREAMLIT ИНТЕРФЕЙС
# ============================================================================

def main():
    # Сайдбар за настройки
    with st.sidebar:
        st.header("⚙️ Настройки на модела")
        
        # Параметри за обработка
        st.subheader("Предварителна обработка")
        h_min = st.number_input("Минимална промяна във височината (h_min)", 
                              value=0.1, min_value=0.0, step=0.1)
        g_max = st.number_input("Максимален наклон (%)", 
                              value=100.0, min_value=10.0, max_value=200.0, step=5.0)
        v_max = st.number_input("Максимална скорост (m/s)", 
                              value=50.0, min_value=10.0, max_value=100.0, step=5.0)
        
        # Параметри за сегментиране
        st.subheader("Сегментиране")
        segment_duration = st.number_input("Продължителност на сегмент (s)", 
                                         value=5, min_value=1, max_value=30, step=1)
        
        # Параметри за Glide модел
        st.subheader("Плъзгаемост")
        alpha_glide = st.slider("Параметър за омекотяване (α)", 
                              min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        
        # Параметри за CS зони
        st.subheader("CS Зони")
        cs_speed_kmh = st.number_input("Критична скорост (km/h)", 
                                     value=15.0, min_value=5.0, max_value=30.0, step=0.5)
        
        # Зареждане на файлове
        st.subheader("Зареждане на данни")
        uploaded_files = st.file_uploader("Изберете TCX файлове", 
                                        type=['tcx'], 
                                        accept_multiple_files=True)
    
    # Главна секция
    if uploaded_files:
        # Параметри
        params = {
            'h_min': h_min,
            'g_max': g_max,
            'v_max': v_max
        }
        
        # Обработване на всеки файл
        all_results = {}
        all_segments = []
        
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.write(f"**Обработване на файл:** {uploaded_file.name}")
            
            # 1. Парсване на TCX
            try:
                df_points = TCXParser.parse_tcx(uploaded_file)
                
                if len(df_points) == 0:
                    st.warning(f"Файлът {uploaded_file.name} не съдържа валидни данни.")
                    continue
                
                # 2. Предварителна обработка
                preprocessor = DataPreprocessor(df_points, params)
                df_clean = preprocessor.preprocess()
                
                # 3. Сегментиране
                segmenter = Segmenter(df_clean, segment_duration)
                segments_df = segmenter.create_segments()
                
                if len(segments_df) == 0:
                    st.warning(f"Не могат да бъдат създадени сегменти от файла {uploaded_file.name}.")
                    continue
                
                # 4. Glide модел
                glide_model = GlideModel(segments_df, alpha_glide)
                segments_df, glide_results = glide_model.run()
                
                # 5. Slope модел
                slope_model = SlopeModel(segments_df)
                segments_df, slope_results = slope_model.run()
                
                # 6. CS Zones модел
                cs_model = CSZonesModel(segments_df, cs_speed_kmh)
                segments_df, cs_results = cs_model.run()
                
                # Запазване на резултатите
                all_results[uploaded_file.name] = {
                    'glide': glide_results,
                    'slope': slope_results,
                    'cs_zones': cs_results,
                    'segments': segments_df
                }
                
                all_segments.append(segments_df.assign(filename=uploaded_file.name))
                
                st.success(f"✅ Файлът {uploaded_file.name} е обработен успешно!")
                
            except Exception as e:
                st.error(f"Грешка при обработка на {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Визуализация на резултатите
        if all_results:
            st.header("📊 Резултати от анализа")
            
            # Избор на файл за детайли
            selected_file = st.selectbox("Изберете файл за детайлен анализ", 
                                       list(all_results.keys()))
            
            if selected_file:
                results = all_results[selected_file]
                segments_df = results['segments']
                
                # Табове за различните визуализации
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Общ преглед", "🎿 Плъзгаемост", "⛰️ Наклон", "🏃 CS Зони"])
                
                with tab1:
                    # Обща статистика
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_time = segments_df['duration'].sum()
                        st.metric("Общо време", f"{total_time/60:.1f} мин")
                    
                    with col2:
                        total_distance = segments_df['distance'].sum()
                        st.metric("Обща дистанция", f"{total_distance/1000:.2f} км")
                    
                    with col3:
                        avg_speed_real = np.average(segments_df['avg_speed'], 
                                                   weights=segments_df['duration'])
                        st.metric("Средна скорост", f"{avg_speed_real*3.6:.1f} км/ч")
                    
                    with col4:
                        avg_speed_final = np.average(segments_df['V_final'], 
                                                    weights=segments_df['duration'])
                        st.metric("Коригирана скорост", f"{avg_speed_final*3.6:.1f} км/ч")
                    
                    # Графика на скоростта във времето
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=segments_df['t_start'],
                        y=segments_df['avg_speed'] * 3.6,
                        mode='lines',
                        name='Реална скорост',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=segments_df['t_start'],
                        y=segments_df['V_final'] * 3.6,
                        mode='lines',
                        name='Коригирана скорост',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title='Скорост по време',
                        xaxis_title='Време',
                        yaxis_title='Скорост (км/ч)',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Резултати от Glide модела
                    if results['glide']:
                        glide_results = results['glide']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Суров индекс", f"{glide_results.get('K_raw', 1):.3f}")
                        
                        with col2:
                            st.metric("Омекотен индекс", f"{glide_results.get('K_soft', 1):.3f}")
                        
                        with col3:
                            st.metric("Брой downhill сегменти", glide_results.get('n_downhill', 0))
                        
                        # Графика на Glide модела
                        if 'a' in glide_results and 'b' in glide_results:
                            fig = go.Figure()
                            
                            # Downhill сегменти
                            downhill_mask = (segments_df['slope_pct'] >= -15) & (segments_df['slope_pct'] <= -5)
                            downhill_data = segments_df[downhill_mask]
                            
                            if len(downhill_data) > 0:
                                fig.add_trace(go.Scatter(
                                    x=downhill_data['slope_pct'],
                                    y=downhill_data['avg_speed'] * 3.6,
                                    mode='markers',
                                    name='Downhill сегменти',
                                    marker=dict(size=8, color='blue')
                                ))
                            
                            # Линеен модел
                            x_range = np.linspace(-15, -5, 50)
                            y_pred = (glide_results['a'] * x_range + glide_results['b']) * 3.6
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name='Glide модел',
                                line=dict(color='red', width=3, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title='Glide модел: Скорост vs Наклон',
                                xaxis_title='Наклон (%)',
                                yaxis_title='Скорост (км/ч)',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Недостатъчно downhill сегменти за анализ на плъзгаемостта.")
                
                with tab3:
                    # Резултати от Slope модела
                    if results['slope']:
                        slope_results = results['slope']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Референтна скорост", f"{slope_results.get('V_flat', 0)*3.6:.1f} км/ч")
                        
                        with col2:
                            st.metric("R-квадрат", f"{slope_results.get('r_squared', 0):.3f}")
                        
                        with col3:
                            st.metric("Сегменти за обучение", slope_results.get('n_training', 0))
                        
                        # Графика на Slope модела
                        if 'c0' in slope_results:
                            fig = go.Figure()
                            
                            # Сегменти за обучение
                            training_mask = (segments_df['slope_pct'] > -3) & (segments_df['slope_pct'] < 10)
                            training_data = segments_df[training_mask]
                            
                            if len(training_data) > 0:
                                delta_V = ((training_data['V_glide'] - slope_results['V_flat']) / 
                                          slope_results['V_flat'] * 100)
                                
                                fig.add_trace(go.Scatter(
                                    x=training_data['slope_pct'],
                                    y=delta_V,
                                    mode='markers',
                                    name='Данни за обучение',
                                    marker=dict(size=8, color='green')
                                ))
                            
                            # Квадратичен модел
                            x_range = np.linspace(-3, 10, 100)
                            y_pred = np.polyval([slope_results['c2'], 
                                               slope_results['c1'], 
                                               slope_results['c0']], x_range)
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_pred,
                                mode='lines',
                                name='ΔV% модел',
                                line=dict(color='orange', width=3)
                            ))
                            
                            fig.update_layout(
                                title='ΔV% модел: Отклонение на скоростта vs Наклон',
                                xaxis_title='Наклон (%)',
                                yaxis_title='ΔV (%)',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Недостатъчно данни за анализ на влиянието на наклона.")
                
                with tab4:
                    # Резултати от CS Zones модела
                    if results['cs_zones']:
                        cs_results = results['cs_zones']
                        zone_stats = cs_results['zone_stats']
                        
                        if not zone_stats.empty:
                            # Кръгова диаграма за разпределение по зони
                            fig = make_subplots(
                                rows=1, cols=2,
                                specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                                subplot_titles=('Разпределение на времето по зони', 
                                               'Средна скорост по зони')
                            )
                            
                            # Кръгова диаграма
                            colors = [cs_results['zone_colors'].get(zone, 'gray') 
                                     for zone in zone_stats['Zone']]
                            
                            fig.add_trace(
                                go.Pie(
                                    labels=zone_stats['Zone'],
                                    values=zone_stats['Percentage (%)'],
                                    hole=0.4,
                                    marker=dict(colors=colors),
                                    textinfo='label+percent',
                                    hoverinfo='label+value+percent'
                                ),
                                row=1, col=1
                            )
                            
                            # Стълбова диаграма
                            fig.add_trace(
                                go.Bar(
                                    x=zone_stats['Zone'],
                                    y=zone_stats['Avg Speed (km/h)'],
                                    marker_color=colors,
                                    text=zone_stats['Avg Speed (km/h)'].round(1),
                                    textposition='auto'
                                ),
                                row=1, col=2
                            )
                            
                            fig.update_layout(
                                height=400,
                                showlegend=False,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Таблица с детайли
                            st.subheader("Детайлна статистика по зони")
                            display_cols = ['Zone', 'Total Time (min)', 'Percentage (%)', 
                                          'Avg Speed (km/h)', 'Segments Count']
                            st.dataframe(zone_stats[display_cols].round(2))
                        else:
                            st.info("Не са намерени данни за CS зоните.")
                    
                    # Информация за CS
                    st.info(f"**Критична скорост (CS):** {cs_speed_kmh:.1f} км/ч")
            
            # Сравнение между файловете (ако има повече от един)
            if len(all_results) > 1:
                st.header("📈 Сравнение между файловете")
                
                # Подготовка на данни за сравнение
                comparison_data = []
                
                for filename, results in all_results.items():
                    segments_df = results['segments']
                    
                    if len(segments_df) > 0:
                        avg_speed_real = np.average(segments_df['avg_speed'], 
                                                   weights=segments_df['duration'])
                        avg_speed_final = np.average(segments_df['V_final'], 
                                                    weights=segments_df['duration'])
                        
                        comparison_data.append({
                            'Файл': filename,
                            'Време (мин)': segments_df['duration'].sum() / 60,
                            'Дистанция (км)': segments_df['distance'].sum() / 1000,
                            'Средна скорост (км/ч)': avg_speed_real * 3.6,
                            'Коригирана скорост (км/ч)': avg_speed_final * 3.6,
                            'Брой сегменти': len(segments_df)
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.round(2))
                    
                    # Графика за сравнение
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Файл'],
                        y=comparison_df['Средна скорост (км/ч)'],
                        name='Реална скорост',
                        marker_color='blue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Файл'],
                        y=comparison_df['Коригирана скорост (км/ч)'],
                        name='Коригирана скорост',
                        marker_color='red'
                    ))
                    
                    fig.update_layout(
                        title='Сравнение на скоростите между файловете',
                        xaxis_title='Файл',
                        yaxis_title='Скорост (км/ч)',
                        barmode='group',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Експорт на резултатите
            st.header("💾 Експорт на резултатите")
            
            if st.button("📥 Изтегли всички резултати като CSV"):
                # Комбиниране на всички сегменти
                if all_segments:
                    combined_df = pd.concat(all_segments, ignore_index=True)
                    
                    # Конвертиране към CSV
                    csv = combined_df.to_csv(index=False, encoding='utf-8-sig')
                    
                    # Сваляне
                    st.download_button(
                        label="Натиснете за сваляне",
                        data=csv,
                        file_name="ski_analysis_results.csv",
                        mime="text/csv"
                    )
    
    else:
        # Начален екран
        st.markdown("""
        ## 🎯 Добре дошли в onFlows Ski Analysis
        
        Това приложение анализира ски-бягане активности чрез три последователни модела:
        
        1. **🎿 Ski Glide Dynamics** - оценка и корекция на плъзгаемостта
        2. **⛰️ Slope Influence** - елиминиране на влиянието на наклона
        3. **🏃 CS Zones** - разпределение на натоварването по физиологични зони
        
        ### 📋 Как да използвате приложението:
        
        1. **Заредете TCX файлове** (един или повече) от вашата ски-бягане активност
        2. **Конфигурирайте параметрите** в левия панел
        3. **Разгледайте резултатите** в различните табове
        4. **Сравнете различните активности** (ако сте заредили повече от един файл)
        
        ### 🔧 Поддържани TCX формати:
        - Garmin устройства
        - Strava експорти
        - Други съвместими TCX файлове
        
        ### ⚠️ Важно:
        - Приложението обработва само валидни точки с GPS координати и височина
        - Препоръчително е всяка активност да има поне 10-15 минути данни
        - Критичната скорост (CS) трябва да бъде определена предварително
        """)
        
        # Примерни параметри
        with st.expander("🔍 Вижте примерни параметри на модела"):
            st.markdown("""
            **Предварителна обработка:**
            - h_min = 0.1 m (минимална промяна във височината)
            - g_max = 100% (максимален допустим наклон)
            - v_max = 50 m/s (максимална допустима скорост)
            
            **Сегментиране:**
            - Продължителност на сегмент = 5 секунди
            
            **Плъзгаемост:**
            - α = 0.5 (коефициент за омекотяване)
            - Downhill диапазон = -15% до -5%
            
            **CS Зони:**
            - Критична скорост = 15 км/ч (по подразбиране)
            - Зони дефинирани спрямо CS: Z1 (0-80%), Z2 (80-90%), Z3 (90-100%),
              Z4 (100-105%), Z5 (105-115%), Z6 (>115%)
            """)

if __name__ == "__main__":
    main()