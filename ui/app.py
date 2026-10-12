import json
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from core.camera import Camera
from core.landmarks import Landmarks
from core.features import OutputRouter, feature, raw_valid
from core.labels import LABELS
from core.dataset import save_batch, read_batch
from core.model_io import train, evaluate, load_model
from network.arena_client import ArenaClient

ROOT=Path(__file__).resolve().parents[1]
CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

class App:
    def __init__(self,config):
        self.config=config;self.root=tk.Tk();self.root.title('手勢訓練器');self.root.geometry('1150x850')
        self.router=OutputRouter();self.camera=None;self.landmarks=None;self.model=None;self.client=None
        self.running=threading.Event();self.image_lock=threading.Lock();self.latest=None;self.notice='開啟鏡頭預覽；正規化與KNN必做，濾波選做。'
        self.algorithm_status={'normalization':False,'knn':False,'filter':False};self.recording=False;self.samples=[];self.labels=[];self.record_start=0;self.last_record=0;self.feature_count=0
        self.filter_enabled=False;self.mode='preview';self.raw_result=None;self.record_label=0;self.record_batch='batch-01';self.record_purpose='train';self.worker=None
        self.capture_reason='鏡頭尚未開啟';self.hand_status='尚未偵測'
        self.normalization_reason=''
        self.sent_availability=None
        self.pair_busy=False;self.batch_paths={}
        self.tool_notifications=queue.Queue()
        main=ttk.Frame(self.root,padding=15);main.pack(fill='both',expand=True)
        ttk.Label(main,text='手勢訓練器',font=('Microsoft JhengHei',22,'bold')).pack(anchor='w',pady=(0,10))
        self.tabs=ttk.Notebook(main);self.tabs.pack(fill='both',expand=True)
        preview=ttk.Frame(self.tabs,padding=12);connect=ttk.Frame(self.tabs,padding=12);data=ttk.Frame(self.tabs,padding=12)
        self.tabs.add(preview,text='即時辨識');self.tabs.add(data,text='採集/訓練');self.tabs.add(connect,text='配對')
        bar=ttk.Frame(preview);bar.pack(fill='x');self.camera_index=tk.StringVar(value='0');ttk.Label(bar,text='鏡頭索引').pack(side='left');ttk.Combobox(bar,textvariable=self.camera_index,values=['0','1','2','3'],width=5).pack(side='left',padx=8)
        ttk.Button(bar,text='開啟鏡頭',command=self.start_camera).pack(side='left');ttk.Button(bar,text='停止鏡頭',command=self.stop_camera).pack(side='left',padx=8)
        self.mode_var=tk.StringVar(value='preview');ttk.Combobox(bar,textvariable=self.mode_var,values=['preview','recognize'],state='readonly',width=12).pack(side='left');self.mode_var.trace_add('write',lambda *_:setattr(self,'mode',self.mode_var.get()))
        self.filter_var=tk.BooleanVar(value=False);ttk.Checkbutton(bar,text='使用學員濾波（選做）',variable=self.filter_var,command=self.change_filter).pack(side='left',padx=8)
        self.details=ttk.Label(preview,text='',wraplength=1000);self.details.pack(side='bottom',fill='x')
        preview_area=ttk.Frame(preview);preview_area.pack(fill='both',expand=True,pady=10)
        self.preview=ttk.Label(preview_area,text='鏡頭尚未開啟',anchor='center');self.preview.place(relwidth=1,relheight=1)
        data_page=data
        data=ttk.Frame(data_page);data.pack(side='left',fill='y',padx=(0,16))
        capture=ttk.LabelFrame(data_page,text='採集即時預覽',padding=10);capture.pack(side='right',fill='both',expand=True)
        self.capture_status=ttk.Label(capture,text='',wraplength=420);self.capture_status.pack(anchor='w',pady=5)
        ttk.Label(capture,text='可錄左手、右手或雙手；同一類保持相同手勢與手數，每次只對應一個動作。',wraplength=420).pack(side='bottom',anchor='w',pady=(10,0))
        capture_area=ttk.Frame(capture);capture_area.pack(fill='both',expand=True)
        self.capture_preview=ttk.Label(capture_area,text='請先到「即時辨識」開啟鏡頭',anchor='center')
        self.capture_preview.place(relwidth=1,relheight=1)
        self.class_var=tk.StringVar(value=LABELS[0]);self.purpose_var=tk.StringVar(value='train')
        for title,var,values in [('錄製類別',self.class_var,LABELS[:10]),('用途',self.purpose_var,['train','test'])]:
            ttk.Label(data,text=title).pack(anchor='w');ttk.Combobox(data,textvariable=var,values=values,state='readonly').pack(anchor='w',pady=5)
        record_bar=ttk.Frame(data);record_bar.pack(anchor='w',pady=5)
        ttk.Button(record_bar,text='開始錄製',command=self.begin_record).pack(side='left')
        ttk.Label(record_bar,text='倒數三秒後開始',foreground='black').pack(side='left',padx=10)
        ttk.Button(data,text='停止錄製',command=self.save_record).pack(anchor='w',pady=5)
        self.counts=ttk.Label(data,text='有效 0 筆');self.counts.pack(anchor='w',pady=5)
        ttk.Label(data,text='錄製資料').pack(anchor='w',pady=10)
        self.files=ttk.Treeview(data,columns=('label','purpose','count','time'),show='headings',height=8,selectmode='browse')
        for key,title,width in [('label','類別',150),('purpose','用途',55),('count','筆數',50),('time','錄製時間',130)]:
            self.files.heading(key,text=title);self.files.column(key,width=width,stretch=False)
        self.files.pack(anchor='w');ttk.Button(data,text='刪除所選資料',command=self.delete_batch).pack(anchor='w',pady=5)
        ttk.Button(data,text='訓練',command=lambda:self.background(self.train,'訓練')).pack(anchor='w',pady=5)
        ttk.Button(data,text='測試',command=lambda:self.background(self.evaluate,'測試')).pack(anchor='w',pady=5)
        ttk.Button(data,text='載入KNN模型',command=lambda:self.background(self.load,'載入KNN模型')).pack(anchor='w',pady=5)
        self.server_var=tk.StringVar(value='http://127.0.0.1:3000');self.room_var=tk.StringVar();self.pair_var=tk.StringVar()
        for title,var in [('主機網址（其他筆電使用主機區網IP）',self.server_var),('房間碼（選填，單人留空）',self.room_var),('一次性配對碼',self.pair_var)]:
            ttk.Label(connect,text=title).pack(anchor='w',pady=5);ttk.Entry(connect,textvariable=var,width=60).pack(anchor='w')
        self.pair_button=ttk.Button(connect,text='配對',command=self.toggle_pair);self.pair_button.pack(anchor='w',pady=10)
        self.root.protocol('WM_DELETE_WINDOW',self.close);self.root.after(50,self.render);self.refresh_files()

    def background(self,fn,operation=None):
        def run():
            try:
                result=fn()
                if operation:self.tool_notifications.put(('info',f'{operation}完成',result))
            except Exception as exc:
                self.notice=f'工具提示：{exc}（算法TODO不等於安裝失敗）'
                if operation:self.tool_notifications.put(('error',f'{operation}失敗',str(exc)))
        threading.Thread(target=run,daemon=True).start()

    def start_camera(self):
        if self.camera:return
        index=int(self.camera_index.get())
        def start():
            try:
                self.camera=Camera(index);self.landmarks=Landmarks(swap_handedness=self.config.get('swap_handedness',False));self.running.set();self.worker=threading.Thread(target=self.process,daemon=True);self.worker.start();self.notice=f'鏡頭實際設定：{self.camera.actual}'
            except Exception as exc:
                self.notice=str(exc);self.stop_camera()
        self.background(start)

    def process(self):
        last=0;previous=time.monotonic()
        while self.running.is_set():
            frame,timestamp=self.camera.latest()
            if frame is None or timestamp==last:
                if frame is None:self.router.update(tracked=False)
                time.sleep(.01);continue
            last=timestamp;start=time.monotonic()
            try:
                sample=self.landmarks.detect(frame,timestamp,self.camera.actual);sample.batch=self.record_batch
                tracked,reason=raw_valid(sample,self.config);result=None;pred=None
                self.capture_reason=reason
                self.hand_status='、'.join(name for name,hand in [('左手',sample.left),('右手',sample.right)] if hand is not None) or '無手（或左右手判定可信度不足）'
                self.normalization_reason=''
                try:
                    result=feature(sample,self.config)
                    if result.valid:self.algorithm_status['normalization']=True
                    else:
                        self.normalization_reason=result.reason or '正規化未接受此樣本'
                        reason=self.normalization_reason
                except Exception as exc:
                    self.algorithm_status['normalization']=False;reason=str(exc)
                    self.normalization_reason=reason
                if self.mode=='recognize' and result and result.valid:
                    if self.model is None:reason='請訓練或載入個人KNN模型'
                    else:
                        pred=self.model.predict(result.features);self.algorithm_status['knn']=True
                self.raw_result=pred
                self.router.update(pred,tracked=tracked and pred is not None,enabled=self.filter_enabled)
                if self.router.error:reason=self.router.error;self.algorithm_status['filter']=False
                elif self.filter_enabled and pred:self.algorithm_status['filter']=True
                self.sync_availability()
                now=time.monotonic()
                if self.recording and now>=self.record_start and now-self.last_record>=.1 and tracked:
                    self.samples.append(sample);self.labels.append(self.record_label);self.last_record=now
                    if result and result.valid:self.feature_count+=1
                for hand in (sample.left,sample.right):
                    if hand is None:continue
                    points=[(int(p[0]*sample.width),int(p[1]*sample.height)) for p in hand]
                    for a,b in CONNECTIONS:cv2.line(frame,points[a],points[b],(90,220,160),2)
                    for point in points:cv2.circle(frame,point,4,(100,180,255),-1)
                fps=1/max(1e-6,now-previous);previous=now
                self.notice=f'{reason or "追蹤正常"} · FPS {fps:.1f} · 辨識 {(now-start)*1000:.1f}ms · 安全輸出 {self.router.read()}'
                with self.image_lock:self.latest=frame
            except Exception as exc:
                self.router.update(tracked=False);self.notice=str(exc)
                self.algorithm_status['normalization']=False;self.algorithm_status['knn']=False
                self.sent_availability=None
                if self.client:
                    try:self.client.availability(normalization=False,knn=False)
                    except Exception:pass

    def stop_camera(self):
        self.running.clear()
        if self.worker and self.worker is not threading.current_thread():self.worker.join(timeout=2)
        if self.camera:self.camera.close();self.camera=None
        if self.landmarks:self.landmarks.close();self.landmarks=None
        with self.image_lock:self.latest=None
        self.preview.config(image='',text='鏡頭已停止')
        self.capture_preview.config(image='',text='請先到「即時辨識」開啟鏡頭')
        self.router.update(tracked=False)
        self.capture_reason='鏡頭已停止';self.hand_status='尚未偵測'
        self.normalization_reason=''

    def change_filter(self):
        self.filter_enabled=self.filter_var.get();self.config['filter_enabled']=self.filter_enabled
        try:self.router.clear(self.filter_enabled)
        except Exception as exc:self.notice=str(exc)
        if self.client:self.background(lambda:self.client.availability(**self.flags()))

    def flags(self):
        return {**self.algorithm_status,'filterEnabled':self.filter_enabled}

    def sync_availability(self):
        client=self.client
        if not client or not client.token:return
        flags=self.flags();signature=(client,tuple(flags.items()))
        if signature==self.sent_availability:return
        try:
            result=client.availability(**flags)
            if result is not None:self.sent_availability=signature
        except Exception as exc:
            self.notice=f'算法狀態同步失敗：{exc}'

    def begin_record(self):
        if not self.camera:messagebox.showinfo('請開鏡頭','先在關鍵點頁開啟鏡頭。');return
        if self.recording:messagebox.showinfo('錄製中','先停止並保存。');return
        self.record_purpose=self.purpose_var.get();self.record_label=LABELS.index(self.class_var.get());self.samples=[];self.labels=[];self.feature_count=0;self.record_start=time.monotonic()+3;self.last_record=0
        if self.record_label==10:self.record_purpose='test';self.purpose_var.set('test')
        self.record_batch=f'{self.record_purpose}-{time.time_ns()}'
        self.recording=True

    def save_record(self):
        self.recording=False
        if not self.samples:
            messagebox.showinfo('沒有有效樣本',f'本次沒有收進有效樣本。\n目前偵測：{self.hand_status}\n採集狀態：{self.capture_reason or "目前可採集"}\n可使用左手、右手或雙手。\n\n確認入鏡的手有完整關鍵點，倒數結束後「有效」筆數持續增加再保存。\n正規化尚未完成不會阻止原始資料採集。');return
        path=ROOT/'data'/f'{self.record_batch}-{self.record_label}-{time.time_ns()}.npz'
        save_batch(path,list(self.samples),list(self.labels),self.record_purpose,dict(self.config));self.notice=f'已保存 {len(self.samples)} 筆原始關鍵點';self.refresh_files()

    def refresh_files(self):
        self.files.delete(*self.files.get_children());self.batch_paths={}
        entries=[]
        for p in (ROOT/'data').glob('*.npz'):
            try:
                _,labels,meta=read_batch(p)
                label=' / '.join(LABELS[int(v)] for v in sorted(set(labels.tolist())))
                purpose=meta['purpose'];count=len(labels)
            except Exception:
                label='資料無法讀取';purpose='—';count='—'
            entries.append((label,purpose,p,count))
            stamp=time.strftime('%m/%d %H:%M:%S',time.localtime(p.stat().st_mtime))
            self.batch_paths[p.name]=p
        for label,purpose,p,count in sorted(entries,key=lambda v:(v[0],v[1],v[2].name)):
            stamp=time.strftime('%m/%d %H:%M:%S',time.localtime(p.stat().st_mtime))
            self.files.insert('', 'end',iid=p.name,values=(label,purpose,count,stamp))

    def delete_batch(self):
        selected=self.files.selection()
        if selected:
            row=self.files.item(selected[0],'values')
            if messagebox.askyesno('刪除批次',f'確認刪除 {row[0]} / {row[1]} / {row[2]} 筆（{row[3]}）？'):
                p=self.batch_paths[selected[0]];p.unlink();p.with_suffix('.json').unlink(missing_ok=True);self.refresh_files()

    def train(self):
        result=train(list((ROOT/'data').glob('*.npz')),ROOT/'models/knn.npz',self.config);self.load()
        self.notice=json.dumps(result,ensure_ascii=False)
        return f'訓練完成，KNN模型已保存並載入。\n訓練樣本：{result["training_samples"]} 筆\n正規化排除：{result["normalization_rejected"]} 筆'

    def evaluate(self):
        result=evaluate(list((ROOT/'data').glob('*.npz')),ROOT/'models/knn.npz',self.config);(ROOT/'models/evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');self.notice=f'獨立測試報告保存 models/evaluation.json：{result}'
        return f'獨立測試完成。\n測試樣本：{result["samples"]} 筆\n正規化排除：{result["normalization_rejected"]} 筆\n報告已保存至 models/evaluation.json'

    def load(self):
        self.model=load_model(ROOT/'models/knn.npz',self.config);self.algorithm_status['knn']=True;self.notice='KNN模型已載入；接口可用不代表真人準確率通過'
        if self.client:self.client.availability(**self.flags())
        return 'KNN模型已載入，可切換至 recognize 模式進行辨識。'

    def toggle_pair(self):
        if self.pair_busy:return
        if self.client and self.client.token:self.disconnect()
        else:self.connect()

    def connect(self):
        server,room,code=self.server_var.get(),self.room_var.get().upper(),self.pair_var.get()
        self.pair_busy=True
        def run():
            try:
                if self.client:self.client.close()
                self.sent_availability=None
                self.client=ArenaClient(self.router,lambda msg:setattr(self,'notice',msg),lambda:self.router.clear(self.filter_enabled));self.client.connect(server,room,code,'recognizer',self.flags())
            except Exception:
                if self.client:self.client.close();self.client=None
                raise
            finally:self.pair_busy=False
        self.background(run)

    def disconnect(self):
        client=self.client;self.pair_busy=True
        def run():
            try:
                if client:client.close()
                self.client=None;self.sent_availability=None;self.notice='已解除配對'
            finally:self.pair_busy=False
        self.background(run)

    def render(self):
        self.pair_button.config(text='解除配對' if self.client and self.client.token else '配對',state='disabled' if self.pair_busy else 'normal')
        self.counts.config(text=f'有效 {len(self.samples)} 筆')
        expected='左手／右手／雙手皆可，可混合採集於同一模型'
        countdown=f'倒數 {max(0,self.record_start-time.monotonic()):.1f} 秒' if self.recording and time.monotonic()<self.record_start else ('錄製中' if self.recording else self.capture_reason or '可採集原始關鍵點')
        normalization_hint=f'\n正規化提示：{self.normalization_reason}' if self.normalization_reason else ''
        self.capture_status.config(text=f'{expected} · 偵測：{self.hand_status}\n採集狀態：{countdown}{normalization_hint}')
        self.details.config(text=f'正規化：{"可用" if self.algorithm_status["normalization"] else "未完成／無效"} · KNN：{"可用" if self.algorithm_status["knn"] else "未完成／未載入"} · 濾波：{"啟用" if self.filter_enabled else "未啟用（選做）"} · 原分類 {self.raw_result}{normalization_hint}')
        with self.image_lock:frame=self.latest
        if frame is not None:
            for widget in (self.preview,self.capture_preview):
                if not widget.winfo_ismapped():continue
                h,w=frame.shape[:2];scale=min(max(1,widget.winfo_width())/w,max(1,widget.winfo_height())/h,1.5)
                image=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(max(1,int(w*scale)),max(1,int(h*scale)))),cv2.COLOR_BGR2RGB))
                photo=ImageTk.PhotoImage(image);widget.config(image=photo,text='');widget.image=photo
        self.root.after(50,self.render)
        try:
            kind,title,message=self.tool_notifications.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind=='error':messagebox.showerror(title,message,parent=self.root)
            else:messagebox.showinfo(title,message,parent=self.root)

    def close(self):
        self.stop_camera()
        if self.client:self.client.close()
        self.root.destroy()

    def run(self):self.root.mainloop()
