"""Forward selection and scheduling from V-band calibration exposures."""

from datetime import date, timedelta
from math import fsum, isfinite, log10, sqrt
from random import Random
from statistics import median


FEATURES = {
    "exposure_seconds": ("单帧曝光时长", "s", 1, 3600),
    "sky_brightness": ("V 波段天空面亮度", "mag/arcsec²", 14, 24),
    "airmass": ("目标方向大气质量", "无量纲", 1, 4),
    "seeing_arcsec": ("视宁度 FWHM", "角秒", 0.3, 12),
}
RESPONSE = "limiting_mag"
LOG_EXPOSURE = "log_exposure"
CONDITIONS = ("sky_brightness", "airmass", "seeing_arcsec")
FEATURE_SETS = (
    ("原秒数", ("exposure_seconds",)),
    ("对数曝光", (LOG_EXPOSURE,)),
    ("原秒数 + 天况", ("exposure_seconds", *CONDITIONS)),
    ("对数曝光 + 天况", (LOG_EXPOSURE, *CONDITIONS)),
)
BASELINE = "历史极限星等中位数"
PENALTIES = (0.001, 0.01, 0.1, 1.0)
FOLD_COUNT = 3
MIN_TRAIN_NIGHTS = 6
DEFAULT_HOLDOUT_NIGHTS = 5
DEFAULT_VALIDATION_NIGHTS = 2
PLAN_DEFAULTS = {
    "exposure_seconds": 240,
    "sky_brightness": 20.8,
    "airmass": 1.3,
    "seeing_arcsec": 2.0,
}
DEFAULT_TARGET_MAG = 20.5
DEFAULT_RESERVE_MAG = 0.3
COLUMNS = [
    {"key": "section", "label": "统计口径"},
    {"key": "item", "label": "组合、系数项、观测夜或曝光"},
    {"key": "penalty", "label": "岭惩罚 α"},
    {"key": "mae", "label": "绝对误差（mag）"},
    {"key": "actual", "label": "实测极限星等或目标星等（mag）"},
    {"key": "estimate", "label": "预测极限星等（mag）"},
    {"key": "coefficient", "label": "模型系数"},
    {"key": "note", "label": "说明或系数单位"},
]


def _add_log_feature(row):
    row[LOG_EXPOSURE] = log10(row["exposure_seconds"] / 60)


def seed() -> list[dict]:
    start = date(2026, 9, 1)
    rng = Random(91026)
    rows = []
    exposures = (40, 75, 150, 240, 480, 720)
    for night in range(24):
        observed_on = start + timedelta(days=night + night // 3)
        sky_base = (19.2, 19.9, 20.9, 21.3, 20.5, 21.5, 19.6, 20.8)[night % 8]
        seeing_base = (1.2, 1.7, 2.5, 1.9, 2.9, 1.5, 2.0)[night % 7]
        zero_point_drift = rng.uniform(-0.07, 0.07)
        for slot in range(5):
            seconds = exposures[(night + slot) % len(exposures)]
            sky = round(sky_base + rng.uniform(-0.18, 0.18), 2)
            airmass = round(1.05 + 0.25 * slot + rng.uniform(0, 0.14), 2)
            seeing = round(seeing_base + rng.uniform(-0.18, 0.18), 2)
            depth = (20.4 + 1.25 * log10(seconds / 60) + 0.5 * (sky - 20.5)
                     - 0.24 * (airmass - 1.2) - 2.5 * log10(seeing / 2)
                     + zero_point_drift + rng.uniform(-0.10, 0.10))
            rows.append({
                "exposure_code": f"YJ-V-{observed_on:%y%m%d}-{slot + 1:02d}",
                "observed_on": observed_on.isoformat(),
                "exposure_seconds": seconds,
                "sky_brightness": sky,
                "airmass": airmass,
                "seeing_arcsec": seeing,
                RESPONSE: round(depth, 3),
            })
    return rows


def _number(value: object, label: str, lower: float, upper: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}须为 {lower} 至 {upper} 之间的有限数值。") from None
    if not isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{label}须为 {lower} 至 {upper} 之间的有限数值。")
    return number


def _validated_row(row: dict) -> dict:
    code = row.get("exposure_code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("校准曝光编号不能为空。")
    observed_on = row.get("observed_on")
    try:
        if date.fromisoformat(observed_on).isoformat() != observed_on:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("观测夜日期须为有效日期，格式为 YYYY-MM-DD。") from None
    result = {"exposure_code": code.strip(), "observed_on": observed_on}
    for key, (label, unit, lower, upper) in FEATURES.items():
        result[key] = _number(row.get(key), f"{label}（{unit}）", lower, upper)
    result[RESPONSE] = _number(row.get(RESPONSE), "实测 V 波段极限星等（mag）", 5, 30)
    _add_log_feature(result)
    return result


def validate(row: dict) -> None:
    _validated_row(row)


def _night_count(value: object, label: str, maximum: int) -> int:
    count = _number(value, label, 1, maximum)
    if not count.is_integer():
        raise ValueError(f"{label}须为整数。")
    return int(count)


def _dot(left: list[float], right: list[float]) -> float:
    return fsum(a * b for a, b in zip(left, right))


def _ridge_solution(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    # The positive ridge penalty makes the Gram system positive definite.
    size = len(rhs)
    lower = [[0.0] * size for _ in rhs]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - _dot(lower[i][:j], lower[j][:j])
            lower[i][j] = sqrt(value) if i == j else value / lower[j][j]
    forward = []
    for i in range(size):
        forward.append((rhs[i] - _dot(lower[i][:i], forward)) / lower[i][i])
    solution = [0.0] * size
    for i in reversed(range(size)):
        tail = fsum(lower[j][i] * solution[j] for j in range(i + 1, size))
        solution[i] = (forward[i] - tail) / lower[i][i]
    return solution


def _fit(rows: list[dict], features: tuple[str, ...], penalty: float) -> tuple[float, tuple[float, ...]]:
    n = len(rows)
    target = [row[RESPONSE] for row in rows]
    if not features:
        return median(target), ()
    target_mean = fsum(target) / n
    centered_target = [value - target_mean for value in target]
    means = []
    scales = []
    columns = []
    for key in features:
        values = [row[key] for row in rows]
        origin = values[0]
        offsets = [value - origin for value in values]
        mean_offset = fsum(offsets) / n
        centered = [offset - mean_offset for offset in offsets]
        peak = max(abs(value) for value in centered)
        means.append(origin + mean_offset)
        if peak == 0:
            scales.append(1.0)
            columns.append(centered)
            continue
        normalized = [value / peak for value in centered]
        rms = sqrt(_dot(normalized, normalized) / n)
        # Subnormal telemetry variation has no practical measurement meaning.
        scale = peak * rms
        if scale < 1e-12:
            scales.append(1.0)
            columns.append([0.0] * n)
        else:
            scales.append(scale)
            columns.append([value / rms for value in normalized])

    gram = [[_dot(left, right) / n for right in columns] for left in columns]
    rhs = [_dot(column, centered_target) / n for column in columns]
    for i in range(len(features)):
        gram[i][i] += penalty
    standardized = _ridge_solution(gram, rhs)
    slopes = tuple(coefficient / scale for coefficient, scale in zip(standardized, scales))
    intercept = target_mean - fsum(slope * mean for slope, mean in zip(slopes, means))
    return intercept, slopes


def _predict(row: dict, features: tuple[str, ...], model: tuple[float, tuple[float, ...]]) -> float:
    intercept, slopes = model
    return fsum([intercept, *(row[key] * slope for key, slope in zip(features, slopes))])


def _errors(rows, features, model):
    return [abs(row[RESPONSE] - _predict(row, features, model)) for row in rows]


def _rank(folds):
    candidates = [(BASELINE, (), 0.0)]
    candidates.extend((name, features, penalty) for name, features in FEATURE_SETS for penalty in PENALTIES)
    ranking = []
    for name, features, penalty in candidates:
        errors = []
        for train, validation in folds:
            errors.extend(_errors(validation, features, _fit(train, features, penalty)))
        ranking.append((fsum(errors) / len(errors), name, features, penalty))
    return sorted(ranking, key=lambda item: (round(item[0], 10), len(item[2]), -item[3], item[1]))


def _schedule(plan, target, reserve, rows, features, penalty, beats_baseline):
    outside = [
        label for key, (label, *_) in FEATURES.items()
        if not min(row[key] for row in rows) <= plan[key] <= max(row[key] for row in rows)
    ]
    if outside:
        return None, None, f"先拍校准星场，{'、'.join(outside)}超出历史校准范围"
    estimate = _predict(plan, features, _fit(rows, features, penalty))
    slack = estimate - target
    if not beats_baseline:
        return estimate, slack, "先拍校准星场，留出表现未优于历史中位数"
    if estimate >= target + reserve:
        return estimate, slack, "可排程，预测深度达到目标及预留余量"
    if slack >= 0:
        return estimate, slack, "先拍校准星场，目标接近预测极限"
    return estimate, slack, "暂缓当前方案，改用更深的曝光方案或等待天况改善"


def _metric(label, value, unit=None):
    result = {"label": label, "value": value}
    if unit:
        result["unit"] = unit
    return result


def analyze(rows: list[dict], parameters: dict) -> dict:
    holdout_nights = _night_count(parameters.get("holdout_nights", DEFAULT_HOLDOUT_NIGHTS), "最终留出夜数", 30)
    validation_nights = _night_count(parameters.get("validation_nights", DEFAULT_VALIDATION_NIGHTS),
                                     "每折验证夜数", 10)
    target = _number(parameters.get("target_mag", DEFAULT_TARGET_MAG), "计划目标 V 星等", 5, 30)
    reserve = _number(parameters.get("reserve_mag", DEFAULT_RESERVE_MAG), "排程预留余量", 0, 3)
    plan = {
        key: _number(parameters.get(f"plan_{key}", PLAN_DEFAULTS[key]), f"计划{label}", lower, upper)
        for key, (label, _, lower, upper) in FEATURES.items()
    }
    _add_log_feature(plan)
    by_night = {}
    seen = set()
    for record in rows:
        row = _validated_row(record)
        code = row["exposure_code"]
        if code in seen:
            raise ValueError(f"校准曝光 {code} 重复，请保留每帧一条校准结果。")
        seen.add(code)
        by_night.setdefault(row["observed_on"], []).append(row)

    nights = sorted(by_night)
    minimum = MIN_TRAIN_NIGHTS + FOLD_COUNT * validation_nights + holdout_nights
    if len(nights) < minimum:
        raise ValueError(f"当前参数至少需要 {minimum} 个不同观测夜，现有 {len(nights)} 夜。")
    for group in by_night.values():
        group.sort(key=lambda row: row["exposure_code"])
    ordered = [row for night in nights for row in by_night[night]]
    development_nights = nights[:-holdout_nights]
    final_nights = nights[-holdout_nights:]
    development = [row for night in development_nights for row in by_night[night]]
    first_validation = len(development_nights) - FOLD_COUNT * validation_nights
    folds = []
    for begin in range(first_validation, len(development_nights), validation_nights):
        train = [row for night in development_nights[:begin] for row in by_night[night]]
        validation = [row for night in development_nights[begin:begin + validation_nights] for row in by_night[night]]
        folds.append((train, validation))

    ranking = _rank(folds)
    cv_mae, name, features, penalty = ranking[0]
    model = _fit(development, features, penalty)
    baseline = _fit(development, (), 0)
    result_rows = [
        {"section": "前向验证·入选" if rank == 1 else "前向验证", "item": candidate,
         "penalty": strength if keys else None, "mae": round(error, 4), "note": f"第 {rank} 名"}
        for rank, (error, candidate, keys, strength) in enumerate(ranking, 1)
    ]
    intercept, slopes = model
    result_rows.append({"section": "留出评估模型系数", "item": "截距", "coefficient": round(intercept, 6), "note": "mag"})
    for key, slope in zip(features, slopes):
        if key == LOG_EXPOSURE:
            label, coefficient_unit = "log₁₀(曝光秒数 / 60)", "mag（曝光 ×10）"
        else:
            label, unit = FEATURES[key][:2]
            coefficient_unit = f"mag / {unit}"
        result_rows.append({"section": "留出评估模型系数", "item": label, "coefficient": round(slope, 6),
                            "note": coefficient_unit})

    holdout_errors = []
    baseline_errors = []
    for night in final_nights:
        group = by_night[night]
        errors = _errors(group, features, model)
        holdout_errors.extend(errors)
        baseline_errors.extend(_errors(group, (), baseline))
        result_rows.append({"section": "留出夜误差", "item": night, "mae": round(fsum(errors) / len(group), 4)})
        for row, error in zip(group, errors):
            result_rows.append({"section": "留出校准曝光", "item": row["exposure_code"], "mae": round(error, 4),
                                "actual": row[RESPONSE], "estimate": round(_predict(row, features, model), 4)})
    holdout_mae = round(fsum(holdout_errors) / len(holdout_errors), 4)
    baseline_mae = round(fsum(baseline_errors) / len(baseline_errors), 4)
    beats_baseline = holdout_mae < baseline_mae
    estimate, slack, decision = _schedule(plan, target, reserve, ordered, features, penalty, beats_baseline)
    result_rows.insert(0, {"section": "计划判定", "item": decision, "actual": target,
                           "estimate": round(estimate, 4) if estimate is not None else None,
                           "note": f"预留 {reserve:g} mag"})
    planning_metrics = []
    if estimate is not None:
        planning_metrics = [_metric("计划预测极限星等", round(estimate, 4), "mag"),
                            _metric("计划星等余量", round(slack, 4), "mag")]
    return {
        "headline": decision,
        "summary": (
            f"三折按夜前向验证选出{name}，验证 MAE 为 {cv_mae:.4f} mag。"
            f"{final_nights[0]} 至 {final_nights[-1]} 的 {holdout_nights} 个留出夜中，"
            f"入选方案 MAE 为 {holdout_mae:.4f} mag，历史中位数基线为 {baseline_mae:.4f} mag。"
            f"排程估算以全部 {len(nights)} 夜的同配置校准数据为依据。"
            "极限星等数值越大，可探测的点源越暗。排程仅适用于同一设备的 V 波段单帧 5σ 口径。"
        ),
        "metrics": [
            _metric("入选方案", name),
            _metric("前向验证 MAE", round(cv_mae, 4), "mag"),
            _metric("最终留出 MAE", holdout_mae, "mag"),
            _metric("历史中位数基线 MAE", baseline_mae, "mag"),
            _metric("留出校准曝光", len(holdout_errors), "帧"),
            *planning_metrics,
        ],
        "columns": COLUMNS,
        "rows": [{"penalty": None, "mae": None, "actual": None, "estimate": None, "coefficient": None,
                  "note": None, **row} for row in result_rows],
        "chart": {"label": "同一留出集的平均绝对误差", "unit": "mag", "points": [
            {"label": "入选方案", "value": holdout_mae},
            {"label": "历史中位数基线", "value": baseline_mae},
        ]},
    }
