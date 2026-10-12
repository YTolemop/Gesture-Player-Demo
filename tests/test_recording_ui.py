import threading
from types import SimpleNamespace

import numpy as np

from core.types import RawHandSample
from ui.app import App


def recording(monkeypatch,tmp_path):
    app=App.__new__(App)
    hand=np.zeros((21,3),dtype=np.float32);hand[9]=[.1,.1,0]
    app.samples=[RawHandSample(None,hand,640,480,0)];app.labels=[0]
    app.record_lock=threading.Lock();app.recording=True;app.record_pending=True
    app.record_batch='test-recording';app.record_label=0;app.record_purpose='train';app.config={}
    app.stop_record_button=SimpleNamespace(config=lambda **_:None)
    app.refresh_files=lambda:None
    app.hand_status='右手';app.capture_reason=''
    monkeypatch.setattr('ui.app.ROOT',tmp_path)
    return app


def test_repeated_stop_saves_only_once(monkeypatch,tmp_path):
    app=recording(monkeypatch,tmp_path)
    app.save_record();app.save_record();app.save_record()
    assert len(list((tmp_path/'data').glob('*.npz')))==1
    assert not app.recording and not app.record_pending


def test_stop_without_samples_does_not_save_or_repeat_dialog(monkeypatch,tmp_path):
    app=recording(monkeypatch,tmp_path);app.samples=[];app.labels=[]
    messages=[]
    monkeypatch.setattr('ui.app.messagebox.showinfo',lambda *args:messages.append(args))
    app.save_record();app.save_record()
    assert not (tmp_path/'data').exists() and len(messages)==1


def test_failed_save_preserves_batch_for_retry(monkeypatch,tmp_path):
    app=recording(monkeypatch,tmp_path)
    from core.dataset import save_batch
    attempts=[]
    def save(*args):
        attempts.append(args)
        if len(attempts)==1:raise OSError('cannot write')
        save_batch(*args)
    monkeypatch.setattr('ui.app.save_batch',save)
    monkeypatch.setattr('ui.app.messagebox.showerror',lambda *_:None)
    app.save_record()
    assert app.record_pending and not app.recording and len(app.samples)==1
    app.save_record();app.save_record()
    assert len(attempts)==2 and len(list((tmp_path/'data').glob('*.npz')))==1


def test_training_actions_remain_visible_when_resized(monkeypatch,tmp_path):
    import tkinter as tk
    root=tk.Tk();root.withdraw()
    monkeypatch.setattr('ui.app.tk.Tk',lambda:root)
    monkeypatch.setattr('ui.app.ROOT',tmp_path)
    app=App({})
    try:
        app.tabs.select(1)
        for width,height in ((1150,850),(800,600),(720,480)):
            root.geometry(f'{width}x{height}+10000+10000');root.deiconify();root.update()
            for button in app.training_buttons:
                assert button.winfo_ismapped()
                assert 0<=button.winfo_rooty()-root.winfo_rooty()
                assert button.winfo_rooty()+button.winfo_height()<=root.winfo_rooty()+root.winfo_height()
                assert button.winfo_rootx()+button.winfo_width()<=root.winfo_rootx()+root.winfo_width()
        assert app.capture_controls.bbox('all')[3]>app.capture_controls.winfo_height()
        app.capture_controls.yview_moveto(1)
        assert app.capture_controls.yview()[0]>0
    finally:
        app.close()
