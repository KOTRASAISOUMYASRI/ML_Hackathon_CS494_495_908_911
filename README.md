# ML_Hackathon_CS494_495_908_911
## Member 1 dataset pipeline

See [the model feature contract](FEATURE_CONTRACT.md) for the training schema.
Install the project requirements and provide one development partition at a time.
Extract `Student folder.zip` and pass its `train` and `student_test` folders
separately. Each trip's files share a basename and directory.

```sh
pip install -r requirements.txt
python dataset_exploration.py "Student folder/train" --labeled --output outputs/train
python dataset_exploration.py "Student folder/student_test" --labeled --output outputs/student_test
python dataset_exploration.py "Student folder/train" --labeled --serve
```

Open http://127.0.0.1:8001 for the development exploration page.
The export produces candidate and observation CSVs, per-trip EDA in JSON and CSV,
EDA plots, and an interactive Folium map for the first trip. Supply separate output folders
and keep the raw dataset and generated artifacts outside Git. Never pass the
reserved evaluator partition to the exploration page or labeled export.
Inference callers can load the evaluator input with `with_reference=False` and
build features with `labeled=False` during final evaluation.

### Dataset API and exploration page integration

The standalone app exposes `GET /api/data/trips` and
`GET /api/data/features?trip_id=<id>`. Pass a trip ID exactly as returned by
the inventory. The response pairs stable candidate IDs with ordered feature
column names, candidate rows, and cleaned observation records. An unknown ID
returns 404. The API skips reference files even when the development page
displays the reference route.

In the shared FastAPI app, include the data router for the intended partition
and mount the exploration page for development trips. These modules can be
used independently.

```python
from fastapi import FastAPI
from data_api import create_data_router
from dataset_exploration import create_app

app = FastAPI()
development_dir = "Student folder/student_test"
app.include_router(create_data_router(development_dir))
app.mount("/eda", create_app(development_dir, labeled=True))
```

For the final evaluator app, use the router with the evaluator input partition
when authorized for the live demo. Mount any labeled exploration page only
against development data. Server-side evaluation owns reference access.
In the combined inference endpoint, call the loader and feature builder
directly if the model, HMM, and data pipeline are running in the same process.

### Dataset exploration preview

![PathSnap Folium map of student-test trip 00000000](docs/figures/student-test-folium.png)

The map shows raw GPS observations in blue, the road network in gray, and the
supplied reference route in green. The [dataset report](DATASET_REPORT.md#report-figures)
includes the training and student-test EDA plots.
