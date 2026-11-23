import json

from sqlalchemy import func, select

from src.domain.analysis import seed
from src.models.base import Base, Session, engine
from src.models.workflow import AnalysisRun, Observation, Review, Schedule
from src.services.observations import clean_observation, content_key


def _version(code, revision):
    return code, int(revision or 1)


def seed_data():
    Base.metadata.create_all(engine)
    rows = [clean_observation(row) for row in seed()]
    with Session.begin() as db:
        keys = set(db.scalars(select(Observation.source_key)))
        # Also treat an exposure version as present if any stored payload matches,
        # so re-seeding an older ledger without revision metadata stays idempotent.
        versions = {_version(item.get('exposure_code'), item.get('revision'))
                    for item in db.scalars(select(Observation.payload))}
        for row in rows:
            key = content_key(row)
            version = _version(row.get('exposure_code'), row.get('revision'))
            if key in keys or version in versions:
                continue
            db.add(Observation(source_key=key, payload=row))
            keys.add(key)
            versions.add(version)
        db.flush()
        return {model.__tablename__: db.scalar(select(func.count(model.id)))
                for model in (Observation, AnalysisRun, Review, Schedule)}


if __name__ == '__main__':
    print(json.dumps(seed_data(), sort_keys=True))
