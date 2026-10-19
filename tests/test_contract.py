import pytest
from main import parser
from network.commands import Commands
from student.pipeline import Pipeline, train


def test_channels():
    assert Commands("MOVE_LEFT", "BLOCK").payload() == {
        "left": "MOVE_LEFT", "right": "BLOCK", "both": "NONE"}
    assert Commands(both="JUMP").both == "JUMP"
    with pytest.raises(ValueError):
        Commands(left="BLOCK", both="JUMP")
    with pytest.raises(ValueError):
        Commands(right="custom gesture")


def test_demo_implements_required_algorithms():
    assert Pipeline.algorithms["normalization"]
    assert Pipeline.algorithms["knn"]
    assert Pipeline.algorithms["filter"]


def test_terminal_contract():
    args = parser().parse_args(["record", "--label", "wave", "--channel", "left", "--seconds", "2"])
    assert args.channel == "left" and args.seconds == 2
    assert parser().parse_args(["camera", "--preview"]).preview
    args = parser().parse_args(["run", "--server", "http://localhost:3000", "--pair-code", "000012"])
    assert args.pair_code == "000012"
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser().parse_args(["camera", "--seconds", value])
