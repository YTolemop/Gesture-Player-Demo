import pytest
from pathlib import Path
import re

from core.features import OutputRouter
from core.types import Prediction
from student.filtering import StudentFilter


@pytest.fixture(autouse=True)
def use_readme_recipe_when_student_filter_is_unfilled(monkeypatch):
    try:
        StudentFilter().reset()
    except NotImplementedError:
        # Validate the tutorial without completing the learner's assignment.
        readme=(Path(__file__).parents[1]/'README.md').read_text(encoding='utf-8')
        block=next(block for block in re.findall(r'```python\n(.*?)```',readme,re.S) if 'class StudentFilter:' in block)
        namespace={}
        exec(block,namespace)
        monkeypatch.setitem(globals(),'StudentFilter',namespace['StudentFilter'])
        monkeypatch.setattr('core.features.StudentFilter',namespace['StudentFilter'])


def prediction(label='JUMP',vote=.8,valid=True):
    return Prediction(label,valid,0.1,vote)


def test_strong_gesture_confirms_on_second_camera_frame():
    f=StudentFilter()
    assert f.update(prediction(),0)=='NONE'
    assert f.update(prediction(),1/30)=='JUMP'
    assert f.update(prediction(),2/30)=='JUMP'


def test_lower_vote_gesture_requires_longer_confirmation():
    f=StudentFilter()
    assert f.update(prediction(vote=.6),0)=='NONE'
    assert f.update(prediction(vote=.6),1/30)=='NONE'
    assert f.update(prediction(vote=.6),2/30)=='JUMP'


def test_one_strong_vote_does_not_skip_confirmation_of_weak_candidate():
    f=StudentFilter()
    assert f.update(prediction(vote=.6),0)=='NONE'
    assert f.update(prediction(vote=1),1/30)=='NONE'


def test_short_valid_glitch_does_not_interrupt_or_retrigger_stable_action():
    f=StudentFilter()
    assert f.update(prediction(),0)=='NONE'
    assert f.update(prediction(),.033)=='JUMP'
    assert f.update(prediction('BLOCK'),.066)=='JUMP'
    assert f.update(prediction(),.099)=='JUMP'
    assert f.update(prediction('BLOCK'),.132)=='JUMP'
    assert f.update(prediction('BLOCK'),.165)=='BLOCK'


def test_changing_candidates_cannot_keep_old_movement_active():
    f=StudentFilter()
    f.update(prediction('MOVE_LEFT'),0)
    assert f.update(prediction('MOVE_LEFT'),.033)=='MOVE_LEFT'
    assert f.update(prediction('JUMP'),.066)=='MOVE_LEFT'
    assert f.update(prediction('BLOCK'),.099)=='MOVE_LEFT'
    assert f.update(prediction('CROUCH'),.132)=='NONE'


@pytest.mark.parametrize('result',[prediction('NONE'),prediction(valid=False)])
def test_unknown_prediction_immediately_stops_and_clears_history(result):
    f=StudentFilter();f.update(prediction(),0);f.update(prediction(),.033)
    assert f.update(result,.04)=='NONE'
    assert f.update(prediction(),.05)=='NONE'
    assert f.update(prediction(),.083)=='JUMP'


@pytest.mark.parametrize('next_time',[.033,0,1])
def test_repeated_timestamp_clock_reset_or_gap_requires_fresh_evidence(next_time):
    f=StudentFilter();f.update(prediction(),0);f.update(prediction(),.033)
    assert f.update(prediction(),next_time)=='NONE'


def test_no_hand_router_bypasses_filter_and_stops_immediately(monkeypatch):
    now=[0]
    monkeypatch.setattr('core.features.time.monotonic',lambda:now[0])
    router=OutputRouter()
    assert router.update(prediction('MOVE_LEFT'),tracked=True,enabled=True)=='NONE'
    now[0]=.033
    assert router.update(prediction('MOVE_LEFT'),tracked=True,enabled=True)=='MOVE_LEFT'
    now[0]=.04
    assert router.update(tracked=False,enabled=True)=='NONE'
    assert router.read()=='NONE'
    now[0]=.05
    assert router.update(prediction('MOVE_LEFT'),tracked=True,enabled=True)=='NONE'
