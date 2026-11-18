# Two-laptop deployment

Install the repository requirements in each laptop's Python environment.

## Laptop 1: API and data

Use the real IP of Laptop 2 in the allowed UI origin if its browser opens the
UI by LAN IP. If Laptop 2 opens `localhost:8080`, the default allowed origin
already includes it. Scheme, hostname and port must match the browser address.

```powershell
$env:PATHSNAP_DATA_ROOT = 'C:/path/to/Student folder'
$env:PATHSNAP_UI_ORIGINS = 'http://localhost:8080,http://127.0.0.1:8080,http://LAPTOP_2_IP:8080'
python -m uvicorn route_recovery.main:app --host 0.0.0.0 --port 8000
```

Keep that terminal open: it records incoming requests, trip IDs, match counts
and inference time. Allow inbound TCP port 8000 on the private network if
Windows Firewall prompts. Use `ipconfig` to identify Laptop 1's Wi-Fi IPv4 address.
No evaluator labels are sent to the UI or used as prediction inputs.

## Laptop 2: UI

```powershell
python -m uvicorn route_recovery.ui_server:app --host 127.0.0.1 --port 8080
```

Open http://localhost:8080. Enter `http://LAPTOP_1_IP:8000` in **Backend API
address** and click Connect. Select a trip and reconstruct it. The address is
remembered locally by the browser. Laptop 2 does not need raw trip files or the
trained model to serve the UI.

Confirm all of the following on the actual two devices:

1. Laptop 2 loads the ten student-test options from Laptop 1.
2. Laptop 1 logs the POST request and selected trip.
3. Laptop 2 shows matched/unmatched counts, runtime, sections and break reasons.
4. Map controls toggle road context, GPS, route and breaks.
5. A wrong API address produces a visible connection error; reconnect restores
   operation. Both devices need internet for Folium assets and map tiles.

Two separate local ports (UI 8080, API 8010), browser reconstruction, visible
metrics and break details, failed-connection errors, recovery after reconnect,
and CORS preflight have been tested in development. The API logged the
selected real trip and successful POST /visualize response. Physical
two-laptop Wi-Fi/firewall behaviour must still be verified on your devices.
The page aborts waiting after three minutes; this does not cancel server work.

## Route diagnostics

```powershell
python -m route_recovery.check_trips student_test --diagnose --output reports/route_break_diagnostics.json
```

Diagnostics use the same candidates and decoder scores as ordinary inference.
They inspect directed reachability and unconstrained minimum path distance at
breaks. They neither change predictions nor read reference routes.
