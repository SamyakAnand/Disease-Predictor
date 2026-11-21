import importlib
from pathlib import Path

import pytest

from healthapp import web_security


class FakePreprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, features):
        self.calls += 1
        return features


class FakeModel:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def predict(self, features):
        self.calls += 1
        if self.error:
            raise self.error
        return [0]


# Import the five entry points without unpickling repository artifacts in tests.
_original_loader = web_security.load_pickle
web_security.load_pickle = lambda _path: None
try:
    APP_MODULES = {
        "Cancer": importlib.import_module("healthapp.Cancer.app"),
        "Diabetes": importlib.import_module("healthapp.Diabetes.app"),
        "Heart": importlib.import_module("healthapp.Heart.app"),
        "Kidney": importlib.import_module("healthapp.Kidney.app"),
        "Liver": importlib.import_module("healthapp.Liver.app"),
    }
finally:
    web_security.load_pickle = _original_loader


@pytest.fixture(params=list(APP_MODULES), ids=list(APP_MODULES))
def legacy_app(request):
    app = APP_MODULES[request.param].app
    model = FakeModel()
    preprocessor = FakePreprocessor()
    app.config.update(
        TESTING=True,
        MODEL=model,
        PREPROCESSOR=preprocessor,
    )
    return app, model, preprocessor


def valid_payload(app):
    return {name: "1" for name in app.config["FEATURE_NAMES"]}


def test_home_page_uses_shared_template_and_disables_caching(legacy_app):
    app, _model, _preprocessor = legacy_app
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert "no-store" in response.headers["Cache-Control"]
    assert "disease=" in response.get_data(as_text=True)


def test_valid_prediction_is_request_scoped_and_has_no_cors_or_storage(
    legacy_app, monkeypatch, tmp_path
):
    app, model, preprocessor = legacy_app
    monkeypatch.chdir(tmp_path)

    response = app.test_client().post(
        "/predict",
        data=valid_payload(app),
        headers={"Origin": "https://untrusted.example"},
    )

    assert response.status_code == 200
    assert "Low Risk" in response.get_data(as_text=True)
    assert model.calls == 1
    assert preprocessor.calls == 1
    assert "no-store" in response.headers["Cache-Control"]
    assert "Access-Control-Allow-Origin" not in response.headers
    assert list(tmp_path.rglob("results.txt")) == []


def test_missing_field_is_rejected_before_model_inference(legacy_app):
    app, model, preprocessor = legacy_app
    payload = valid_payload(app)
    payload.pop(app.config["FEATURE_NAMES"][0])

    response = app.test_client().post("/predict", data=payload)

    assert response.status_code == 400
    assert model.calls == 0
    assert preprocessor.calls == 0


@pytest.mark.parametrize("bad_value", ["", "not-a-number", "NaN", "inf", "-inf"])
def test_invalid_numeric_values_are_rejected(legacy_app, bad_value):
    app, model, preprocessor = legacy_app
    payload = valid_payload(app)
    payload[app.config["FEATURE_NAMES"][0]] = bad_value

    response = app.test_client().post("/predict", data=payload)

    assert response.status_code == 400
    assert model.calls == 0
    assert preprocessor.calls == 0


def test_model_failure_is_generic_and_does_not_leak_exception_text(legacy_app):
    app, _model, preprocessor = legacy_app
    failing_model = FakeModel(error=RuntimeError("private backend detail"))
    app.config["MODEL"] = failing_model

    response = app.test_client().post("/predict", data=valid_payload(app))

    body = response.get_data(as_text=True)
    assert response.status_code == 503
    assert "temporarily unavailable" in body
    assert "private backend detail" not in body
    assert failing_model.calls == 1
    assert preprocessor.calls == 1


def test_shared_results_history_is_disabled(legacy_app):
    app, _model, _preprocessor = legacy_app
    response = app.test_client().get("/results")

    assert response.status_code == 410
    assert "disabled" in response.get_data(as_text=True).lower()
    assert "no-store" in response.headers["Cache-Control"]


def test_artifact_file_is_closed_after_loading(tmp_path, monkeypatch):
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_bytes(b"test artifact")
    opened_files = []

    def fake_pickle_load(file_handle):
        opened_files.append(file_handle)
        assert not file_handle.closed
        return {"loaded": True}

    monkeypatch.setattr(web_security.pickle, "load", fake_pickle_load)

    assert web_security.load_pickle(artifact_path) == {"loaded": True}
    assert len(opened_files) == 1
    assert opened_files[0].closed
