import csv
import hashlib
import io
import json

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.config.project import SPEC
from src.domain import analysis
from src.models.workflow import AuditEvent, Observation
from src.services.errors import BusinessError
from src.services.validation import validated


MAX_OBSERVATIONS = 5000


def clean_observation(raw):
    """Validate a received exposure version, folding version metadata in.

    Two records for the same exposure code are distinct versions keyed by
    ``revision``; identical retries collapse to one ledger row downstream via
    the content key.
    """
    row = validated(SPEC['fields'], raw)
    try:
        if hasattr(analysis, 'validated_revision'):
            return analysis.validated_revision(row)
        if hasattr(analysis, 'validate'):
            analysis.validate(row)
    except ValueError as exc:
        raise BusinessError(str(exc)) from exc
    return row


def content_key(row):
    """Stable identity of a received version, independent of its receipt clock.

    ``received_at`` is excluded so a retried correction deduplicates instead of
    creating a second version. The first acceptance keeps its receipt instant.
    """
    canonical = {key: row.get(key) for key in (
        'exposure_code', 'observed_on', 'exposure_seconds', 'sky_brightness',
        'airmass', 'seeing_arcsec', 'limiting_mag', 'revision')}
    canonical['revision'] = int(canonical.get('revision') or 1)
    return 'data:' + hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_csv(raw):
    if len(raw) > 512 * 1024:
        raise BusinessError('CSV 文件上限 512 KB', 413)
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
        header = reader.fieldnames or []
        known = {item['key'] for item in SPEC['fields']}
        required = {item['key'] for item in SPEC['fields'] if item.get('required', True)}
        if len(set(header)) != len(header) or not required.issubset(header) or not set(header).issubset(known):
            raise BusinessError('CSV 表头与导入模板不匹配')
        rows = []
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise BusinessError('列数与表头不匹配')
                rows.append(clean_observation({key: value for key, value in row.items()
                                               if value is not None and value != ''}))
            except BusinessError as exc:
                raise BusinessError(f'CSV 第 {reader.line_num} 行，{exc}') from exc
            if len(rows) > 500:
                raise BusinessError('单次导入上限 500 条')
        if not rows:
            raise BusinessError('CSV 里没有观测记录')
        return rows
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BusinessError('请上传 UTF-8 编码的有效 CSV') from exc


def _conflicting_revisions(db, rows):
    """Reject a received version that collides with a different stored payload."""
    codes = {row['exposure_code'] for row in rows}
    if not codes:
        return
    stored = {}
    for item in db.scalars(select(Observation.payload)):
        code = item.get('exposure_code')
        if code in codes:
            stored[(code, int(item.get('revision') or 1))] = item
    for row in rows:
        key = (row['exposure_code'], int(row.get('revision') or 1))
        previous = stored.get(key)
        if previous is not None and content_key(previous) != content_key(row):
            raise BusinessError(
                f'校准曝光 {key[0]} 的修订 {key[1]} 已有不同内容，请用更高修订号提交更正', 409)


def add_observations(db, rows, actor, action):
    keyed = {content_key(row): row for row in rows}
    existing = set(db.scalars(select(Observation.source_key).where(Observation.source_key.in_(keyed))))
    fresh = {key: row for key, row in keyed.items() if key not in existing}
    if db.scalar(select(func.count(Observation.id))) + len(fresh) > MAX_OBSERVATIONS:
        raise BusinessError('当前项目最多保留 5000 条观测记录', 409)
    _conflicting_revisions(db, fresh.values())
    items = [Observation(source_key=key, payload=row) for key, row in fresh.items()]
    db.add_all(items)
    if items:
        corrections = sum(1 for row in fresh.values() if int(row.get('revision') or 1) > 1)
        detail = f'新增 {len(items)} 条观测' + (f'（含 {corrections} 条更正）' if corrections else '')
        db.add(AuditEvent(actor=actor, action=action, detail=detail))
    try:
        db.flush()
    except IntegrityError as exc:
        raise BusinessError('有相同记录刚刚写入，请刷新后重试', 409) from exc
    return items
