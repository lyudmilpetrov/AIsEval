from __future__ import annotations

import csv
import io
from typing import Annotated

import torch
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.datastructures import FormData, UploadFile

router = APIRouter(prefix="/api/Regression", tags=["Regression"])


class CsvPrediction(BaseModel):
    rowIndex: int
    prediction: float


class CsvRegressionResponse(BaseModel):
    framework: str
    model: str
    gpuRequested: bool
    gpuUsed: bool
    trainingRows: int
    testRows: int
    featureCount: int
    predictions: list[CsvPrediction]


class CsvFormatError(ValueError):
    pass


@router.get("/Test", response_class=PlainTextResponse)
def test() -> str:
    return "ping"


@router.post("/Predict", response_model=CsvRegressionResponse)
async def predict(
    request: Request,
    use_gpu: Annotated[bool, Query(alias="UseGPU")] = False,
) -> CsvRegressionResponse | JSONResponse:
    form = await request.form()
    features_file = _find_csv_file(form, "features", "features.csv")
    tests_file = _find_csv_file(form, "tests", "tests.csv")

    if features_file is None or tests_file is None:
        return _bad_request(
            "Upload multipart/form-data files named features.csv and tests.csv, or use form fields named features and tests."
        )

    try:
        training_rows = _read_numeric_csv(
            await features_file.read(), features_file.filename or "features.csv"
        )
        test_rows = _read_numeric_csv(
            await tests_file.read(), tests_file.filename or "tests.csv"
        )
    except CsvFormatError as exc:
        return _bad_request(str(exc))

    if not training_rows:
        return _bad_request("features.csv must contain at least one training row.")
    if not test_rows:
        return _bad_request("tests.csv must contain at least one test row.")
    if len(training_rows[0]) < 2:
        return _bad_request(
            "features.csv must contain one or more feature columns followed by a target column."
        )

    feature_count = len(training_rows[0]) - 1
    if any(len(row) != feature_count + 1 for row in training_rows):
        return _bad_request(
            "Every features.csv row must have the same number of columns."
        )
    if any(len(row) != feature_count for row in test_rows):
        return _bad_request(
            f"Every tests.csv row must contain exactly {feature_count} feature columns."
        )

    # Keep the regression endpoint CPU-only so benchmark runs match the current
    # AiDotNet controller behavior, which reports the GPU request but does not
    # configure GPU acceleration yet.
    device = torch.device("cpu")
    gpu_used = False

    x_train = torch.tensor(
        [row[:feature_count] for row in training_rows],
        dtype=torch.float64,
        device=device,
    )
    y_train = torch.tensor(
        [[row[feature_count]] for row in training_rows],
        dtype=torch.float64,
        device=device,
    )
    x_tests = torch.tensor(test_rows, dtype=torch.float64, device=device)

    predictions = _predict_with_torch_least_squares(x_train, y_train, x_tests)

    return CsvRegressionResponse(
        framework="PyTorch",
        model="torch.linalg.lstsq-linear-regression",
        gpuRequested=use_gpu,
        gpuUsed=gpu_used,
        trainingRows=len(training_rows),
        testRows=len(test_rows),
        featureCount=feature_count,
        predictions=[
            CsvPrediction(rowIndex=index, prediction=value)
            for index, value in enumerate(predictions)
        ],
    )


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message})


def _find_csv_file(
    form: FormData, field_name: str, file_name: str
) -> UploadFile | None:
    uploads: list[tuple[str, UploadFile]] = [
        (key, value)
        for key, value in form.multi_items()
        if isinstance(value, UploadFile)
    ]

    field_name_lower = field_name.lower()
    file_name_lower = file_name.lower()

    return (
        next(
            (upload for key, upload in uploads if key.lower() == field_name_lower), None
        )
        or next(
            (upload for key, upload in uploads if key.lower() == file_name_lower), None
        )
        or next(
            (
                upload
                for _, upload in uploads
                if (upload.filename or "").lower() == file_name_lower
            ),
            None,
        )
    )


def _predict_with_torch_least_squares(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_tests: torch.Tensor,
) -> list[float]:
    train_bias = torch.ones(
        (x_train.shape[0], 1), dtype=x_train.dtype, device=x_train.device
    )
    test_bias = torch.ones(
        (x_tests.shape[0], 1), dtype=x_tests.dtype, device=x_tests.device
    )
    train_design = torch.cat((train_bias, x_train), dim=1)
    test_design = torch.cat((test_bias, x_tests), dim=1)

    solution = torch.linalg.lstsq(train_design, y_train).solution
    predicted = test_design.matmul(solution).squeeze(dim=1)
    return [float(value) for value in predicted.detach().cpu().tolist()]


def _parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def _read_numeric_csv(content: bytes, filename: str) -> list[list[float]]:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows: list[list[float]] = []
    first_data_row = True

    for raw_row in reader:
        if not raw_row or all(not value.strip() for value in raw_row):
            continue

        trimmed_row = [value.strip() for value in raw_row]
        try:
            rows.append([_parse_float(value) for value in trimmed_row])
            first_data_row = False
        except ValueError as exc:
            if first_data_row:
                first_data_row = False
                continue
            raise CsvFormatError(
                f"CSV file '{filename}' contains a non-numeric data row: {','.join(raw_row)}"
            ) from exc

    return rows
