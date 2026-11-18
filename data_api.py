"""Read-only trip inventory and label-free preprocessing for PathSnap."""
from fastapi import APIRouter, HTTPException, Query

from data_loader import discover_trips, load_trip
from feature_engineering import FEATURE_COLUMNS, ID_COLUMNS, build_feature_table


def create_data_router(partition_dir):
    """Attach the configured partition at /api/data, with no reference reads."""
    trips = {files.trip_id: files for files in discover_trips(partition_dir)}
    router = APIRouter(prefix="/api/data", tags=["dataset"])

    @router.get("/trips")
    def list_trips():
        return {"trip_ids": list(trips)}

    @router.get("/features")
    def get_features(trip_id: str = Query(min_length=1)):
        files = trips.get(trip_id)
        if files is None:
            raise HTTPException(status_code=404, detail="Unknown trip ID")
        trip = load_trip(files, with_reference=False)
        result = build_feature_table(trip, labeled=False)
        return {
            "trip_id": trip_id,
            "id_columns": ID_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "candidates": result.candidates.to_dict(orient="records"),
            "observations": result.observations.to_dict(orient="records"),
        }

    return router
