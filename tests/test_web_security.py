import numpy as np
import pytest

import app as web_app


class StubPreprocessor:
    def transform(self, frame):
        return frame


class StubModel:
    def predict(self, frame):
        return np.array([1])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "load_artifacts",
        lambda _disease: (StubModel(), StubPreprocessor()),
    )
    web_app.app.config.update(TESTING=True)
    return web_app.app.test_client()


def valid_form(disease):
    return {
        feature: str(index + 1)
        for index, feature in enumerate(
            web_app.disease_configs[disease]["PREDICTION_FEATURES"]
        )
    }


@pytest.mark.parametrize("disease", sorted(web_app.disease_configs))
def test_valid_prediction_renders_without_persisting_health_inputs(
    client, monkeypatch, tmp_path, disease
):
    monkeypatch.chdir(tmp_path)

    response = client.post(
        f"/predict/{disease}",
        data=valid_form(disease),
    )

    assert response.status_code == 200
    assert "High Risk" in response.get_data(as_text=True)
    assert not (tmp_path / "Results").exists()
    assert response.headers["Cache-Control"].startswith("no-store")


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing", "Please complete every field"),
        ("blank", "Please complete every field"),
        ("nonnumeric", "Enter a valid number"),
        ("nan", "Values must be finite"),
        ("infinity", "Values must be finite"),
    ],
)
def test_invalid_numeric_inputs_are_rejected(
    client, mutation, expected_message
):
    form = valid_form("heart")
    if mutation == "missing":
        form.pop("chol")
    elif mutation == "blank":
        form["chol"] = "   "
    elif mutation == "nonnumeric":
        form["chol"] = "not-a-number"
    elif mutation == "nan":
        form["chol"] = "NaN"
    elif mutation == "infinity":
        form["chol"] = "Infinity"

    response = client.post("/predict/heart", data=form)

    assert response.status_code == 400
    assert expected_message in response.get_data(as_text=True)


def test_invalid_input_is_not_reflected_as_html(client):
    form = valid_form("heart")
    form["chol"] = "<script>alert(1)</script>"

    response = client.post("/predict/heart", data=form)
    body = response.get_data(as_text=True)

    assert response.status_code == 400
    assert "<script>alert(1)</script>" not in body
    assert "Enter a valid number for every field." in body


def test_model_unavailable_returns_generic_service_error(client, monkeypatch):
    monkeypatch.setattr(web_app, "load_artifacts", lambda _disease: (None, None))

    response = client.post("/predict/heart", data=valid_form("heart"))
    body = response.get_data(as_text=True)

    assert response.status_code == 503
    assert "temporarily unavailable" in body
    assert "Traceback" not in body


def test_prediction_exception_does_not_leak_internal_details(client, monkeypatch):
    class BrokenPreprocessor:
        def transform(self, _frame):
            raise RuntimeError("private-model-path")

    monkeypatch.setattr(
        web_app,
        "load_artifacts",
        lambda _disease: (StubModel(), BrokenPreprocessor()),
    )

    response = client.post("/predict/heart", data=valid_form("heart"))
    body = response.get_data(as_text=True)

    assert response.status_code == 500
    assert "could not be completed" in body
    assert "private-model-path" not in body


def test_prediction_history_endpoint_is_disabled(client):
    response = client.get("/results?disease=heart")

    assert response.status_code == 410
    assert "history is disabled" in response.get_data(as_text=True)


def test_server_rendered_app_does_not_enable_cross_origin_access(client):
    response = client.get("/", headers={"Origin": "https://untrusted.example"})

    assert "Access-Control-Allow-Origin" not in response.headers
