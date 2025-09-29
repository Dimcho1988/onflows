# onFlows – MVP (Streamlit)

Минимално работещо приложение:
- Strava OAuth → списък активности → 1 Hz CSV за избрана активност.
- Индекси: TIZ, дневен стрес, ACWR, HR drift.
- Генератор: седмичен план с таргети (HR / %CS / %CP) и адаптация при висок ACWR.
- Настройки: HRmax, CS, CP, зони (от `config.yaml`).

## Инсталация (локално)

```bash
git clone https://github.com/<your-account>/onflows.git
cd onflows
pip install -r requirements.txt
streamlit run streamlit_app.py
