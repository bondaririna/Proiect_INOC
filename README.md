# INOC - Asistent de gatit hands-free

Aplicatie Python pentru control hands-free al YouTube prin gesturi, folosind:

- `MediaPipe` pentru detectia mainilor
- `OpenCV` pentru captura camera si preview

## Cerinte

- Windows 10/11
- Python 3.10 sau 3.11
- Webcam functionala

## Rulare rapida (PowerShell)

```powershell
.\scripts\run.ps1
```

Scriptul:

1. creeaza `.venv` daca nu exista
2. activeaza mediul virtual
3. instaleaza dependintele
4. ruleaza `src/main.py`

## Rulare manuala

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\main.py
```

## Gesturi implementate

- `Swipe right` cu toate degetele ridicate -> `Right Arrow` (YouTube `+5s`)
- `Swipe left` cu toate degetele ridicate -> `Left Arrow` (YouTube `-5s`)
- `Open palm` (o mana, mentinut scurt) -> `K` (YouTube `pause/play`) — activ si cand timerul este vizibil
- `2 palme deschise` -> toggle timer (afisare/ascundere)
- `2 degete ridicate` (cand timerul este vizibil) -> `+5 minute` la timer
- `Pumn inchis` (cand timerul este vizibil) -> start/pause countdown
- `Index up motion` (miscare in sus cu aratatorul) -> `Volume Up`
- `Index down hold` (aratator orientat in jos, mentinut) -> `Volume Down`
- `Crossed index fingers` (X cu doua aratatoare, doua maini) -> `Ctrl+W` (inchide tab-ul curent)
- `Tasta Q` (de la tastatura) -> inchide aplicatia

## Afisare in fereastra camerei

- In preview se afiseaza doar timerul (`TIMER HH:MM:SS`), cand este vizibil.

## Configurare `.env`

### Camera si detectie

- `CAMERA_INDEX`
- `FRAME_WIDTH`
- `FRAME_HEIGHT`
- `DISPLAY_WINDOW_WIDTH`
- `DISPLAY_WINDOW_HEIGHT`
- `MIN_DETECTION_CONFIDENCE`
- `MIN_TRACKING_CONFIDENCE`

### Swipe left/right

- `SWIPE_RIGHT_MIN_DELTA_X`
- `SWIPE_RIGHT_WINDOW_SECONDS`
- `SWIPE_RIGHT_COOLDOWN_SECONDS`
- `SWIPE_RIGHT_MIN_SAMPLES`

### Open palm (pause/play)

- `PAUSE_OPEN_PALM_MIN_SPAN_X`
- `OPEN_PALM_HOLD_SECONDS`
- `OPEN_PALM_COOLDOWN_SECONDS`

### Timer

- `TIMER_ACTIVATE_TWO_PALMS_HOLD_SECONDS`
- `TIMER_ACTIVATE_TWO_PALMS_COOLDOWN_SECONDS`
- `OPEN_PALM_MIN_SPAN_X` (pentru validarea celor 2 palme deschise)
- `TIMER_ADD_FIVE_HOLD_SECONDS`
- `TIMER_ADD_FIVE_COOLDOWN_SECONDS`
- `TIMER_START_FIST_HOLD_SECONDS`
- `TIMER_START_FIST_COOLDOWN_SECONDS`

### Volum

- `VOLUME_UP_MIN_DELTA_Y`
- `VOLUME_UP_WINDOW_SECONDS`
- `VOLUME_UP_COOLDOWN_SECONDS`
- `VOLUME_UP_MIN_SAMPLES`
- `VOLUME_DOWN_HOLD_SECONDS`
- `VOLUME_DOWN_COOLDOWN_SECONDS`

### Inchidere tab (gest X)

- `CLOSE_X_HOLD_SECONDS`
- `CLOSE_X_COOLDOWN_SECONDS`

## Observatii

- Pentru gesturile care trimit taste, browserul cu YouTube trebuie sa poata primi input de tastatura.
- Comenzile de volum folosesc taste media Windows (controleaza volumul sistemului).
