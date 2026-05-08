from __future__ import annotations

import csv
import io
from typing import Annotated

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/Regression", tags=["Regression"])


class CsvPrediction(BaseModel):
    rowIndex: int
    prediction: float


class CsvRegressionResponse(BaseModel):
    framework: str
    model: str
    gpuRequested: bool
    gpuUsed: bool
    device: str
    trainingRows: int
    testRows: int
    featureCount: int
    predictions: list[CsvPrediction]


@router.get("/Test", response_class=PlainTextResponse)
def test() -> str:
    return "ping"


@router.post("/Predict", response_model=CsvRegressionResponse)
async def predict(
    features: Annotated[
        UploadFile,
        File(description="features.csv with feature columns followed by the target column"),
    ],
    tests: Annotated[UploadFile, File(description="tests.csv with feature columns only")],
    use_gpu: Annotated[bool, Query(alias="UseGPU")] = False,
) -> CsvRegressionResponse:
    training_rows = _read_numeric_csv(await features.read(), features.filename or "features.csv")
    test_rows = _read_numeric_csv(await tests.read(), tests.filename or "tests.csv")

    if not training_rows:
        raise HTTPException(status_code=400, detail="features.csv must contain at least one training row.")
    if not test_rows:
        raise HTTPException(status_code=400, detail="tests.csv must contain at least one test row.")
    if len(training_rows[0]) < 2:
        raise HTTPException(
            status_code=400,
            detail="features.csv must contain one or more feature columns followed by a target column.",
        )

    feature_count = len(training_rows[0]) - 1
    if any(len(row) != feature_count + 1 for row in training_rows):
        raise HTTPException(status_code=400, detail="Every features.csv row must have the same number of columns.")
    if any(len(row) != feature_count for row in test_rows):
        raise HTTPException(
            status_code=400,
            detail=f"Every tests.csv row must contain exactly {feature_count} feature columns.",
        )

    gpu_used = bool(use_gpu and torch.cuda.is_available())
    device = torch.device("cuda:0" if gpu_used else "cpu")

    x_train = torch.tensor([row[:feature_count] for row in training_rows], dtype=torch.float64, device=device)
    y_train = torch.tensor([[row[feature_count]] for row in training_rows], dtype=torch.float64, device=device)
    x_tests = torch.tensor(test_rows, dtype=torch.float64, device=device)

    predictions = _predict_with_torch_least_squares(x_train, y_train, x_tests)

    return CsvRegressionResponse(
        framework="PyTorch",
        model="torch.linalg.lstsq-linear-regression",
        gpuRequested=use_gpu,
        gpuUsed=gpu_used,
        device=str(device),
        trainingRows=len(training_rows),
        testRows=len(test_rows),
        featureCount=feature_count,
        predictions=[CsvPrediction(rowIndex=index, prediction=value) for index, value in enumerate(predictions)],
    )


def _predict_with_torch_least_squares(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_tests: torch.Tensor,
) -> list[float]:
    train_bias = torch.ones((x_train.shape[0], 1), dtype=x_train.dtype, device=x_train.device)
    test_bias = torch.ones((x_tests.shape[0], 1), dtype=x_tests.dtype, device=x_tests.device)
    train_design = torch.cat((train_bias, x_train), dim=1)
    test_design = torch.cat((test_bias, x_tests), dim=1)

    solution = torch.linalg.lstsq(train_design, y_train).solution
    predicted = test_design.matmul(solution).squeeze(dim=1)
    return [float(value) for value in predicted.detach().cpu().tolist()]


def _read_numeric_csv(content: bytes, filename: str) -> list[list[float]]:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows: list[list[float]] = []
    first_data_row = True

    for raw_row in reader:
        if not raw_row or all(not value.strip() for value in raw_row):
            continue

        try:
            rows.append([float(value.strip()) for value in raw_row])
            first_data_row = False
        except ValueError as exc:
            if first_data_row:
                first_data_row = False
                continue
            raise HTTPException(
                status_code=400,
                detail=f"CSV file '{filename}' contains a non-numeric data row: {raw_row}",
            ) from exc

    return rows
