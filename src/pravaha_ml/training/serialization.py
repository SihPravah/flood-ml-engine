from pathlib import Path
import pickle

from pravaha_ml.models.baseline import (
    BaselineRiskModel,
)


def save_model(
    model: BaselineRiskModel,
    path: str | Path,
) -> None:
    if not model.is_fitted:
        raise RuntimeError(
            "Cannot save an unfitted model."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as file:
        pickle.dump(
            model,
            file,
        )


def load_model(
    path: str | Path,
) -> BaselineRiskModel:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    with path.open("rb") as file:
        model = pickle.load(file)

    if not isinstance(
        model,
        BaselineRiskModel,
    ):
        raise TypeError(
            "Serialized object is not "
            "a BaselineRiskModel."
        )

    return model