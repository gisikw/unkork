from click.testing import CliRunner

from unkork.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "regression codec" in result.output


def test_generate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--voices-dir" in result.output


def test_train_help():
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--epochs" in result.output


def test_predict_help():
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_refine_help():
    runner = CliRunner()
    result = runner.invoke(main, ["refine", "--help"])
    assert result.exit_code == 0
    assert "--iterations" in result.output
    assert "--scorer" in result.output
    assert "--mel-weight" in result.output
