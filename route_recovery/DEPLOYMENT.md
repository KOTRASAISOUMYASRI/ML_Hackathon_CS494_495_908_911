# Connect the two laptops

Laptop 1 runs the API, datasets, optimized HistGradientBoosting and HMM.
Laptop 2 serves the UI and receives results over HTTP. Connect both to the same
Wi-Fi/hotspot with device-to-device communication enabled. Internet is needed
for map tiles and Folium assets. Run commands from the repository root.

## 1. Laptop 1: backend

The existing project .venv is already verified. On a fresh machine create it
first: `py -3.14 -m venv .venv` (Python 3.14 was tested locally).

```powershell
.\.venv\Scripts\python.exe -m pip install -r route_recovery/requirements-inference.txt
git lfs pull
powershell -NoProfile -File .\route_recovery\start_api.ps1
```

Keep this terminal open. The server binds 0.0.0.0:8000 so Laptop 2 can reach it.
The dataset defaults to the repo's train/student_test folders. For another parent:

```powershell
powershell -NoProfile -File .\route_recovery\start_api.ps1 -DataRoot 'C:/your/dataset/parent'
```

For the authorized evaluation, provide the evaluator input and reference
directories on Laptop 1. Each directory preserves the trip subfolders and
file basenames. When the organizers provide all four file types together,
pass the supplied directory for both parameters:

```powershell
$evaluator = 'C:/path/to/evaluator_test'
powershell -NoProfile -File .\route_recovery\start_api.ps1 -EvaluatorInput $evaluator -EvaluatorReference $evaluator
```

The trip picker shows only evaluator trips and requests server-side scoring
automatically. The backend reads references after prediction.

The inference environment pins scikit-learn 1.8.0 to match the saved artifacts.
The selected model is outputs/optimization/hist_gradient_boosting_optimized.joblib.
A full LFS pull also downloads the 1.16 GB baseline Random Forest and 531 MB
optimized forest. For only the selected model, use this instead of the full pull:

```powershell
git lfs pull --include="outputs/optimization/hist_gradient_boosting_optimized.joblib"
```

In another terminal run `ipconfig`. Find Laptop 1's Wi-Fi IPv4 address, for
example 192.168.1.10. Its API address is http://192.168.1.10:8000. Replace this
example IP with the actual value in every command below.

Open http://127.0.0.1:8000/health on Laptop 1. The default_model must be
hist_gradient_boosting_optimized and default_model_available true. This checks
file availability; the route check below also loads the model and runs HMM.

## 2. Laptop 2: UI

Use the same latest branch/version as Laptop 1. To clone without unnecessary
model downloads:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --branch main https://github.com/KOTRASAISOUMYASRI/ML_Hackathon_CS494_495_908_911.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd ML_Hackathon_CS494_495_908_911
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r route_recovery/requirements-ui.txt
powershell -NoProfile -File .\route_recovery\start_ui.ps1
```

For an existing clean clone of this branch, use `git pull --ff-only` instead of
cloning again. Preserve local work before switching/updating. Laptop 2 only
needs FastAPI/Uvicorn and UI source; datasets/sklearn/models stay on Laptop 1.

Open **http://localhost:8080** on Laptop 2. In **Backend API address**, enter
Laptop 1's actual API address, e.g. **http://192.168.1.10:8000**, and click
**Connect**. An API address using localhost here would point to Laptop 2 itself.

The allowed UI origins default to http://localhost:8080 and http://127.0.0.1:8080.
If you change the UI port or open it using another hostname/IP, restart Laptop 1
with that exact browser origin. For example (the extra IP below is Laptop 2's):

```powershell
powershell -NoProfile -File .\route_recovery\start_api.ps1 -UiOrigins 'http://localhost:8080,http://127.0.0.1:8080,http://192.168.1.20:8080'
```

## 3. What to demonstrate

1. The selected HGB aggregate panel shows all ten student-test trips: Top-1
   **81.92% alone -> 79.49% with HMM**, a **-2.43 percentage-point** change.
   Continuity rises **85.25% -> 98.27%**. These use derived development labels.
2. In the authorized evaluator run, confirm the picker lists all ten evaluator
   trips. Keep optimized HGB selected, let the evaluator choose a trip, and
   click **Reconstruct route**.
3. Show the current evaluator-trip comparison, accuracy change, runtime, map
   and gaps. The saved aggregate panel is from the student-test partition.
   For rehearsal on development data, use the connection check below.
4. Toggle blue GPS, purple independent selections, green HMM sections, grey
   roads and orange breaks. No artificial line crosses a break.
5. Show Laptop 1's incoming POST /visualize, model/trip, match counts and runtime.

The aggregate panel is separate from the current trip. If the model artifact's
fingerprint changes, stale aggregate results are hidden until verification is
rerun. Reference routes are only read after inference when evaluation is
requested; reference IDs and geometry are not returned to the UI.

## 4. Record receiving-laptop evidence

In another Laptop 2 terminal, from the repo root:

```powershell
.\.venv\Scripts\python.exe -m route_recovery.check_connection --api http://192.168.1.10:8000 --trip 00000052/00000052
```

This checks health, trip discovery, CORS preflight, real route inference and map
reception. It prints received-byte counts and saves reports/laptop2_connection.json
on Laptop 2. Failed checks exit nonzero and save the error. Use --origin if the
browser UI origin differs from localhost:8080. Omit --trip for connectivity
checks without inference/reference evaluation.

Collect the Laptop 2 result screenshot and saved JSON, plus Laptop 1's incoming
request log. Those establish the actual two-device demonstration.

## Troubleshooting

- Cannot open Laptop 1 /health from Laptop 2: check its IPv4/port, API binding
  0.0.0.0, and shared network. Guest Wi-Fi/client isolation may block peers;
  use a shared hotspot if needed.
- Windows Firewall: allow the Python API on the Private network. If an explicit
  rule is needed, run on Laptop 1 in Administrator PowerShell:
  `New-NetFirewallRule -DisplayName 'PathSnap API 8000' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private`.
- /health works but browser Connect fails: match the UI's exact origin using
  -UiOrigins, then restart the API.
- Empty trips: confirm the dataset parent contains the partition folders with
  same-stem .track, .nodes and .arcs files.
- Missing/LFS-pointer model or sklearn mismatch: download the actual selected
  artifact and install requirements-inference.txt on Laptop 1.
- Port busy: close the old server or change -Port; update API address or allowed
  UI origin to match. Both scripts accept -Port.
- No tiles: check internet. Numeric API results do not depend on map tiles.
  The browser's three-minute timeout does not cancel server computation.

## Verification status

Local regression and HTTP checks verify separate UI/API serving, CORS,
selected-model route reception, errors and the comparison panel. They do not
prove physical Wi-Fi/firewall behaviour between two laptops. Complete steps
1-4 on both devices to verify that final physical connection.

The reserved evaluation requires the evaluator input and reference directories
described above. The recorded results use student-test data.
