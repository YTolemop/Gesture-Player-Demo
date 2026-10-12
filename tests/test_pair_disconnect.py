import threading
from types import SimpleNamespace

import pytest
import socketio

from network.arena_client import ArenaClient
from ui.app import App


def client():
    c=ArenaClient.__new__(ArenaClient)
    c.lock=threading.RLock();c.stop=threading.Event()
    c.token='old-token';c.room='ABC123';c.seq=3
    c.context={'sessionId':'session','generation':1};c.pending_preview={'image':'old'};c.last_preview=1
    c.router=SimpleNamespace(clear=lambda:None)
    c.reset=lambda:None;c.on_status=lambda _:None
    c.thread=SimpleNamespace(join=lambda **_:None)
    c.sio=SimpleNamespace(connected=True,namespaces=['/controller'],emit=lambda *_,**__:None,disconnect=lambda:None)
    return c


@pytest.mark.parametrize('reason',[socketio.Client.reason.SERVER_DISCONNECT,socketio.Client.reason.CLIENT_DISCONNECT])
def test_intentional_disconnect_clears_pairing_credentials(reason):
    c=client();assert c.paired
    c._disconnected(reason)
    assert not c.paired and c.token is None and c.room is None
    assert c.context=={} and c.pending_preview is None and c.stop.is_set()
    c._context({'sessionId':'late-session'})
    assert c.context=={}


def test_transport_interruption_shows_unpaired_but_can_resume():
    c=client()
    c._disconnected(socketio.Client.reason.TRANSPORT_ERROR)
    assert not c.paired and c.token=='old-token' and not c.stop.is_set()
    c._context({'sessionId':'resumed-session'})
    assert c.paired


def test_close_clears_pairing_even_without_disconnect_callback():
    c=client();c.close()
    assert not c.paired and c.token is None and c.room is None
    assert c.context=={} and c.pending_preview is None


def test_rejected_resume_discards_stale_token(monkeypatch):
    c=client();c.context={}
    messages=[];c.on_status=messages.append
    class ImmediateThread:
        def __init__(self,target,**_):self.target=target
        def start(self):self.target()
    monkeypatch.setattr('network.arena_client.threading.Thread',ImmediateThread)
    def rejected(*_):raise RuntimeError('INVALID_TOKEN')
    c.call=rejected;c._connected()
    assert c.token is None and not c.paired and c.stop.is_set()
    assert 'INVALID_TOKEN' in messages[-1]


def test_pair_button_action_uses_live_connection_instead_of_stale_token():
    app=App.__new__(App);app.pair_busy=False;app.client=client()
    actions=[];app.connect=lambda:actions.append('connect');app.disconnect=lambda:actions.append('disconnect')
    app.toggle_pair();assert actions==['disconnect']
    app.client._disconnected(socketio.Client.reason.TRANSPORT_ERROR)
    app.toggle_pair();assert actions==['disconnect','connect']
