from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.guard import validate_xyzt_data, validate_xyzt_required_files


def _valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {"store": [1, 1], "item": [1, 1], "sales": [13, 11]},
        index=pd.DatetimeIndex(["2013-01-01", "2013-01-02"], name="date"),
    )
    test = pd.DataFrame(
        {"id": [0, 1], "store": [1, 1], "item": [1, 1]},
        index=pd.DatetimeIndex(["2018-01-01", "2018-01-02"], name="date"),
    )
    sample_submission = pd.DataFrame({"id": [0, 1], "sales": [0, 0]})
    return train, test, sample_submission


class TestXYZTGuard(unittest.TestCase):
    def test_valid_dataset_passes_without_mutation(self) -> None:
        train, test, sample_submission = _valid_frames()
        original_train = train.copy(deep=True)
        original_test = test.copy(deep=True)
        original_submission = sample_submission.copy(deep=True)

        validate_xyzt_data(train, test, sample_submission)

        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(test, original_test)
        pd.testing.assert_frame_equal(sample_submission, original_submission)

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            (raw_dir / "train.csv").touch()

            with self.assertRaisesRegex(FileNotFoundError, "test.csv, sample_submission.csv"):
                validate_xyzt_required_files(raw_dir)

    def test_missing_required_column_raises(self) -> None:
        train, test, sample_submission = _valid_frames()
        test = test.drop(columns=["item"])

        with self.assertRaisesRegex(ValueError, "test columns must be"):
            validate_xyzt_data(train, test, sample_submission)

    def test_duplicate_ids_raise(self) -> None:
        train, test, sample_submission = _valid_frames()
        test.loc[test.index[1], "id"] = 0

        with self.assertRaisesRegex(ValueError, "test contains duplicate IDs"):
            validate_xyzt_data(train, test, sample_submission)

    def test_incorrect_schema_raises(self) -> None:
        train, test, sample_submission = _valid_frames()
        sample_submission = sample_submission[["sales", "id"]]

        with self.assertRaisesRegex(ValueError, "sample_submission columns must be"):
            validate_xyzt_data(train, test, sample_submission)


if __name__ == "__main__":
    unittest.main()
