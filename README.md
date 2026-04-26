# INOC - Asistent de gatit hands-free (Setup)

Acest repository contine doar setup-ul initial pentru aplicatia practica:

- detectie mana cu `MediaPipe`
- captura video/webcam cu `OpenCV`
- schelet pregatit pentru integrarea gesturilor din tema

## Cerinte

- Windows 10/11
- Python 3.10 sau 3.11
- Webcam functionala

## Setup rapid (PowerShell)

Din directorul proiectului:

```powershell
.\scripts\run.ps1
```

Scriptul:

1. creeaza `.venv` daca nu exista
2. activeaza mediul virtual
3. instaleaza dependintele
4. ruleaza aplicatia (`src/main.py`)

## Setup manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\main.py
```

## Stare curenta

Setup-ul include doar infrastructura de start si pipeline de baza pentru detectie mana.
Clasificarea completa a gesturilor (swipe, hold, check sign, etc.) ramane pentru etapa de implementare.
