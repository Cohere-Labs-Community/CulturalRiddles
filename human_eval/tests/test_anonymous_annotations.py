import sys

import pandas as pd
import pytest

from human_eval.validate_humaneval_submission import main


@pytest.mark.parametrize("mutation", ["personal_id", "email_column", "missing_id"])
def test_workbook_accepts_opaque_ids_and_rejects_identity_leaks(tmp_path, monkeypatch, mutation):
    workbook = tmp_path / "Human Eval English_Canada.xlsx"
    frame = pd.DataFrame(
        {
            "Annotator ID": ["annotator_q4x2"],
            "Score (0-3)": [0],
            "Error Type": ["0.1 - Too broad"],
            "Validator Note": [""],
        }
    )

    def save():
        with pd.ExcelWriter(workbook) as writer:
            frame.to_excel(writer, sheet_name="Main", index=False)
            frame.to_excel(writer, sheet_name="Extras", index=False)

    monkeypatch.setattr(sys, "argv", ["validate", str(tmp_path)])
    save()
    assert main() == 0
    if mutation == "personal_id":
        frame["Annotator ID"] = "A person's name"
    elif mutation == "email_column":
        frame["Email"] = "private@example.test"
    else:
        frame = frame.drop(columns="Annotator ID")
    save()
    assert main() == 1
    assert {path.name for path in tmp_path.iterdir()} == {workbook.name}
