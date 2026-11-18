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

See [dataset report notes](DATASET_REPORT.md) for the method and evidence to collect.
