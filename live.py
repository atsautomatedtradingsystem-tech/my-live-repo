# live.py — оновлений для живої симуляції (демо) з non-blocking AE та уніфікованими open/close
import os
from pathlib import Path


# --- Trade notification helper ---
def notify_trade(fig, entry_t, entry_p, side='long', exit_t=None, exit_p=None, trade_id=None):
    """Add clear annotations and a soft shaded region to indicate the trade lifecycle.
    - entry_t, exit_t: datetime-like or x-axis values compatible with plotly's x axis
    - entry_p, exit_p: numeric prices
    - side: 'long' or 'short' (used to color the region)
    - trade_id: optional identifier shown in the annotation
    This helper is defensive and will not raise if fig is not a plotly Figure.
    """
    try:
        import plotly.graph_objs as go
    except Exception:
        # if plotly not present or unavailable, silently skip
        return
    try:
        color = 'green' if (str(side).lower().startswith('l')) else 'red'
        # Add entry annotation (arrow pointing to the price)
        try:
            fig.add_annotation(x=entry_t, y=entry_p,
                               text=f"OPEN{' ' + str(trade_id) if trade_id else ''}\n{str(side).upper()}\n{entry_p}",
                               showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor='rgba(0,0,0,0)', bordercolor=color, borderwidth=1, opacity=0.95)
        except Exception:
            pass
        # If trade closed, add close annotation and a shaded rect between entry and exit to show duration
        if exit_t is not None and exit_p is not None:
            try:
                fig.add_annotation(x=exit_t, y=exit_p,
                                   text=f"CLOSE\n{exit_p}", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor='rgba(0,0,0,0)', bordercolor='black', borderwidth=1, opacity=0.95)
            except Exception:
                pass
            try:
                # shaded full-height rectangle between entry and exit
                fig.add_shape(dict(type='rect', xref='x', yref='paper', x0=entry_t, x1=exit_t, y0=0, y1=1,
                                   fillcolor=color, opacity=0.08, layer='below', line_width=0))
            except Exception:
                pass
            try:
                # PnL summary in the middle top of the shaded region
                pnl = None
                try:
                    pnl = (float(exit_p) - float(entry_p)) if str(side).lower().startswith('l') else (float(entry_p) - float(exit_p))
                except Exception:
                    pnl = None
                if pnl is not None:
                    txt = f"PNL: {pnl:.2f} | Entry: {entry_p} | Exit: {exit_p}"
                else:
                    txt = f"Entry: {entry_p} | Exit: {exit_p}"
                fig.add_annotation(x=entry_t + (exit_t - entry_t) / 2 if hasattr(entry_t, '__add__') else entry_t,
                                   y=1, yref='paper', text=txt, showarrow=False, bgcolor='rgba(0,0,0,0)', bordercolor=color, borderwidth=1, opacity=0.9)
            except Exception:
                pass
        else:
            # for open trades, add a dotted vertical line to show where it started
            try:
                fig.add_shape(dict(type='line', xref='x', yref='paper', x0=entry_t, x1=entry_t, y0=0, y1=1,
                                   line=dict(color=color, width=1, dash='dot')))
            except Exception:
                pass
    except Exception:
        # never let visualization helper break main program
        return


    try:
        # add a subtle circular marker at the entry price for quick visual cue (non-intrusive)
        try:
            fig.add_trace(go.Scatter(x=[entry_t], y=[entry_p], mode='markers',
                                     marker=dict(size=8, symbol='circle', color=color, line=dict(color='white', width=1)),
                                     showlegend=False, hoverinfo='skip'))
        except Exception:
            pass
    except Exception:
        pass



# ------------------ Calibrator & Calibration Data Collection ------------------
import pickle as _pickle
import bisect as _bisect

class SimpleBinnedCalibrator:
    """Fallback calibrator using binning (histogram) mapping from score -> empirical prob"""
    def __init__(self, n_bins=20):
        self.n_bins = n_bins
        self.bins = None
        self.probs = None

    def fit(self, scores, y):
        try:
            # build bins over score range
            scores = [float(s) for s in scores]
            ymin = min(scores) if scores else -1.0
            ymax = max(scores) if scores else 1.0
            if ymin == ymax:
                ymin -= 1.0; ymax += 1.0
            bins = [ymin + (ymax - ymin) * i / self.n_bins for i in range(self.n_bins+1)]
            sums = [0.0]*self.n_bins
            cnts = [0]*self.n_bins
            for sc, label in zip(scores, y):
                idx = min(self.n_bins-1, max(0, int((sc - bins[0]) / (bins[-1]-bins[0]) * self.n_bins)))
                sums[idx] += float(label)
                cnts[idx] += 1
            probs = []
            for i in range(self.n_bins):
                probs.append((sums[i] / cnts[i]) if cnts[i]>0 else 0.5)
            self.bins = bins
            self.probs = probs
        except Exception:
            self.bins = None
            self.probs = None

    def predict_proba(self, scores):
        out = []
        for sc in scores:
            try:
                if self.bins is None or self.probs is None:
                    out.append(0.5)
                    continue
                # find bin
                idx = _bisect.bisect_right(self.bins, float(sc)) - 1
                idx = min(max(0, idx), len(self.probs)-1)
                out.append(self.probs[idx])
            except Exception:
                out.append(0.5)
        return out

class PlattCalibrator:
    """Platt scaling using sklearn LogisticRegression if available, else fallback to simple fit."""
    def __init__(self):
        self.model = None
        self.available = False
        try:
            from sklearn.linear_model import LogisticRegression as _LR
            self._LR = _LR
            self.available = True
        except Exception:
            self.available = False

    def fit(self, scores, y):
        try:
            X = [[float(s)] for s in scores]
            yv = [int(bool(x)) for x in y]
            if self.available:
                clf = self._LR(solver='lbfgs')
                clf.fit(X, yv)
                self.model = clf
            else:
                # fallback: simple linear fit via pseudo-inverse for logit transform approximation
                import math as _math, statistics as _stats
                # compute mean and variance mapping
                mean_s = _stats.mean(scores) if scores else 0.0
                mean_p = _stats.mean(yv) if yv else 0.5
                # store simple intercept/slope heuristic
                self.model = {'mean_s': mean_s, 'mean_p': mean_p}
        except Exception:
            self.model = None

    def predict_proba(self, scores):
        out = []
        try:
            if self.available and hasattr(self.model, 'predict_proba'):
                X = [[float(s)] for s in scores]
                probs = self.model.predict_proba(X)
                for p in probs:
                    out.append(p[1])
                return out
            else:
                # fallback mapping
                for s in scores:
                    try:
                        if self.model is None:
                            out.append(0.5)
                        else:
                            mean_s = self.model.get('mean_s', 0.0)
                            mean_p = self.model.get('mean_p', 0.5)
                            # simple logistic-shaped mapping around mean_s
                            import math as _math
                            val = 1.0 / (1.0 + _math.exp(-0.5*(float(s)-mean_s)))
                            out.append(max(0.01, min(0.99, val)))
                    except Exception:
                        out.append(0.5)
                return out
        except Exception:
            return [0.5]*len(scores)

# container and utilities
def _ensure_calibration_storage():
    try:
        if isinstance(active_state, dict):
            if 'calibration_records' not in active_state:
                active_state['calibration_records'] = []  # list of (score, label)
            if 'calibrators' not in active_state:
                active_state['calibrators'] = {}
    except Exception:
        pass

def record_trade_calibration(tr):
    try:
        if not tr:
            return
        _ensure_calibration_storage()
        score = tr.get('open_score') if isinstance(tr, dict) else None
        if score is None:
            return
        # label: positive pnl (use pnl_scaled or pnl)
        pnl = None
        try:
            pnl = float(tr.get('pnl_scaled') if tr.get('pnl_scaled') is not None else tr.get('pnl') if tr.get('pnl') is not None else 0.0)
        except Exception:
            pnl = 0.0
        label = 1 if pnl > 0 else 0
        active_state['calibration_records'].append((float(score), int(label)))
    except Exception:
        pass

def train_global_calibrator(min_samples=50, method='platt'):
    """Train a global calibrator from collected records and store in active_state['calibrators']['global']"""
    try:
        _ensure_calibration_storage()
        recs = list(active_state.get('calibration_records', []))
        if not recs or len(recs) < min_samples:
            return None
        scores = [r[0] for r in recs]
        labels = [r[1] for r in recs]
        if method == 'platt':
            cal = PlattCalibrator()
        else:
            cal = SimpleBinnedCalibrator(n_bins=30)
        cal.fit(scores, labels)
        # store object in memory
        active_state['calibrators']['global'] = cal
        try:
            log.info("Trained global calibrator on %d samples (method=%s)", len(scores), method)
        except Exception:
            pass
        return cal
    except Exception:
        try:
            log.exception("train_global_calibrator failed")
        except Exception:
            pass
        return None

def predict_calibrated_prob(score, sym=None):
    try:
        _ensure_calibration_storage()
        # prefer symbol-specific then global
        cals = active_state.get('calibrators', {}) or {}
        cal = None
        if sym and sym in cals:
            cal = cals[sym]
        elif 'global' in cals:
            cal = cals['global']
        if cal:
            return float(cal.predict_proba([score])[0])
        # fallback simple mapping
        import math as _math
        try:
            return 1.0 / (1.0 + _math.exp(-float(score)))
        except Exception:
            return 0.5
    except Exception:
        return 0.5

# wrap execute_close_trade to collect calibration labels after close
def _install_close_trade_calibration_wrapper():
    try:
        if 'execute_close_trade' in globals():
            orig = globals()['execute_close_trade']
            def _wrapped(sym, wallet, tr, price, reason='unknown', dry=True):
                res = None
                try:
                    res = orig(sym, wallet, tr, price, reason=reason, dry=dry)
                    # after original, attempt to record calibration if trade closed
                    try:
                        # ensure tr refers to final trade dict
                        final_tr = tr
                        # if orig returns a dict maybe it's the trade
                        if isinstance(res, dict) and 'open_score' in res and ('pnl' in res or 'pnl_scaled' in res):
                            final_tr = res
                        # record if closed (has exit_price or pnl)
                        if final_tr and (final_tr.get('exit_price') is not None or final_tr.get('pnl') is not None or final_tr.get('pnl_scaled') is not None):
                            record_trade_calibration(final_tr)
                    except Exception:
                        pass
                    return res
                except Exception as e:
                    try:
                        log.exception("Wrapped execute_close_trade encountered error: %s", e)
                    except Exception:
                        pass
                    # still try to record
                    try:
                        if tr:
                            record_trade_calibration(tr)
                    except Exception:
                        pass
                    raise
            globals()['_orig_execute_close_trade_for_cal'] = orig
            globals()['execute_close_trade'] = _wrapped
    except Exception:
        try:
            log.exception("Failed to install execute_close_trade calibration wrapper")
        except Exception:
            pass

# attempt install on import
try:
    _install_close_trade_calibration_wrapper()
except Exception:
    pass

# ------------------ end calibrator code ------------------

# --- end helper ---


# --- Immediate startup calibrator training (runs once at import) ---
def _collect_historical_calibration_data():
    scores = []
    labels = []
    try:
        # 1) existing collected records
        if isinstance(active_state, dict):
            recs = active_state.get('calibration_records') or []
            for sc, lab in recs:
                try:
                    scores.append(float(sc)); labels.append(int(lab))
                except Exception:
                    continue
        # 2) scan wallets' past trades for open_score and pnl/exit_price
        try:
            wallets = active_state.get('wallets', {}) if isinstance(active_state, dict) else {}
            for sym, w in (wallets or {}).items():
                try:
                    for tr in (w.get('trades') or []):
                        try:
                            sc = tr.get('open_score') if isinstance(tr, dict) else None
                            if sc is None:
                                continue
                            # only use closed trades (have exit_price or pnl info)
                            if tr.get('exit_price') is None and tr.get('pnl') is None and tr.get('pnl_scaled') is None:
                                continue
                            pnl = None
                            try:
                                pnl = float(tr.get('pnl_scaled') if tr.get('pnl_scaled') is not None else tr.get('pnl') if tr.get('pnl') is not None else 0.0)
                            except Exception:
                                pnl = 0.0
                            scores.append(float(sc)); labels.append(1 if pnl>0 else 0)
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass
        # 3) optional: load calibration_history.csv if present in cwd
        try:
            import csv, io, sys
            p = Path('calibration_history.csv')
            if p.exists():
                with p.open('r', encoding='utf-8') as cf:
                    rdr = csv.DictReader(cf)
                    for r in rdr:
                        try:
                            sc = r.get('score') or r.get('sc') or r.get('s')
                            lab = r.get('label') or r.get('y') or r.get('lbl')
                            if sc is None or lab is None:
                                continue
                            scores.append(float(sc)); labels.append(int(lab))
                        except Exception:
                            continue
        except Exception:
            pass
    except Exception:
        try:
            log.exception('_collect_historical_calibration_data failed')
        except Exception:
            pass
    return scores, labels

def train_calibrator_startup(min_samples=1, method='platt', persist_path='calibrator_global.pkl'):
    """Train a global calibrator immediately from any available historical data.
       If insufficient data found, this function will still create a default PlattCalibrator
       with no training (fallback) so that predict_calibrated_prob will use a consistent object.
    """
    try:
        scores, labels = _collect_historical_calibration_data()
        # ensure lists
        scores = [float(x) for x in scores] if scores else []
        labels = [int(x) for x in labels] if labels else []
        # If too few examples but user requested immediate training from first trade,
        # we will still train if at least min_samples provided. Otherwise create empty calibrator.
        cal = None
        if len(scores) >= min_samples:
            try:
                cal = PlattCalibrator()
                cal.fit(scores, labels)
                try:
                    log.info('Startup calibrator trained on %d samples', len(scores))
                except Exception:
                    pass
            except Exception:
                try:
                    log.exception('Startup PlattCalibrator training failed, falling back to binned')
                except Exception:
                    pass
                try:
                    cal = SimpleBinnedCalibrator(n_bins=20)
                    cal.fit(scores, labels)
                except Exception:
                    cal = None
        # If no calibrator trained, create default PlattCalibrator object (unfitted)
        if cal is None:
            try:
                cal = PlattCalibrator()
            except Exception:
                cal = SimpleBinnedCalibrator(n_bins=20)
        # store into active_state
        try:
            if isinstance(active_state, dict):
                cals = active_state.get('calibrators') or {}
                cals['global'] = cal
                active_state['calibrators'] = cals
        except Exception:
            pass
        # persist to disk for next restarts
        try:
            with open(persist_path, 'wb') as pf:
                _pickle.dump(cal, pf)
        except Exception:
            try:
                log.exception('Failed to persist calibrator to disk')
            except Exception:
                pass
        return cal
    except Exception:
        try:
            log.exception('train_calibrator_startup failed')
        except Exception:
            pass
        try:
            # fallback: ensure global calibrator exists in active_state
            if isinstance(active_state, dict):
                if 'calibrators' not in active_state:
                    active_state['calibrators'] = {}
                if 'global' not in active_state['calibrators']:
                    active_state['calibrators']['global'] = PlattCalibrator()
            return active_state['calibrators']['global']
        except Exception:
            return None

# Run training immediately (non-blocking small), using min_samples=1 per user's request
try:
    _startup_cal = train_calibrator_startup(min_samples=1, method='platt', persist_path='calibrator_global.pkl')
except Exception:
    try:
        log.exception('Startup calibrator training failed')
    except Exception:
        pass

# --- end startup calibrator code ---



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import asyncio
import random
import threading
import logging
import datetime
from logging.config import dictConfig
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import collections
import json
import sqlite3

import requests
import typer
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import ccxt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
try:
    import faiss
    _has_faiss = True
except Exception:
    _has_faiss = False
from tenacity import retry, wait_exponential, stop_after_attempt
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

# --- Assistant minimal helpers: safe fetch, background retrain, order guard ---
import asyncio as _asyncio_helper
try:
    _ORDER_LOCK = _asyncio_helper.Lock()
except Exception:
    _ORDER_LOCK = None

def safe_sync_fetch(symbol, timeframe, limit):
    try:
        return OHLCV.sync_fetch(symbol, timeframe, limit)
    except Exception as _e:
        try:
            log.warning("OHLCV.sync_fetch failed for %s: %s", symbol, _e)
        except Exception:
            print("OHLCV.sync_fetch failed for", symbol, _e)
        return None

async def safe_async_fetch(symbol, timeframe, limit):
    try:
        return await OHLCV.async_fetch(symbol, timeframe, limit)
    except Exception as _e:
        try:
            log.warning("OHLCV.async_fetch failed for %s: %s", symbol, _e)
        except Exception:
            print("OHLCV.async_fetch failed for", symbol, _e)
        return None

def schedule_retrain(df, wnd_min, model_name):
    try:
        loop = _asyncio_helper.get_event_loop()
    except Exception:
        loop = None
    async def _runner():
        try:
            if callable(train_autoencoder):
                res = train_autoencoder(df, wnd_min, model_name)
                try:
                    last_retrain_ts = time.time()
                except Exception:
                    pass
                if _asyncio_helper.iscoroutine(res):
                    await res
        except Exception as e:
            try:
                log.exception("Background retrain failed: %s", e)
            except Exception:
                print("Background retrain failed:", e)
    try:
        if loop and loop.is_running():
            _asyncio_helper.create_task(_runner())
        else:
            import threading
            def _run_in_thread():
                try:
                    _asyncio_helper.run(_runner())
                except Exception as e:
                    try:
                        log.exception("Background retrain run error: %s", e)
                    except Exception:
                        print("Background retrain run error:", e)
            threading.Thread(target=_run_in_thread, daemon=True).start()
    except Exception as e:
        try:
            log.exception("Failed to schedule retrain: %s", e)
        except Exception:
            print("Failed to schedule retrain:", e)

# Safe open wrapper: preserve original if exists and wrap to check bar guards and serialize opens.
try:
    if 'execute_open_trade' in globals() and '_orig_execute_open_trade' not in globals():
        _orig_execute_open_trade = globals()['execute_open_trade']
except Exception:
    _orig_execute_open_trade = globals().get('execute_open_trade')

def _make_safe_open():
    async def _safe_open(*args, **kwargs):
        lock = _ORDER_LOCK
        if lock is not None:
            await lock.acquire()
        try:
            cb = globals().get('current_bar_id')
            if globals().get('last_closed_bar_id') is not None and cb is not None and cb == globals().get('last_closed_bar_id'):
                try:
                    log.info('SafeOpen: skip open because last close on same bar %s', cb)
                except Exception:
                    pass
                return None
            if globals().get('last_opened_bar_id') is not None and cb is not None and cb == globals().get('last_opened_bar_id'):
                try:
                    log.info('SafeOpen: skip open because already opened on bar %s', cb)
                except Exception:
                    pass
                return None
            orig = globals().get('_orig_execute_open_trade')
            if orig is None:
                try:
                    log.warning('SafeOpen: original execute_open_trade missing')
                except Exception:
                    pass
                return None
            try:
                res = orig(*args, **kwargs)
                if _asyncio_helper.iscoroutine(res):
                    res = await res
                try:
                    globals()['last_opened_bar_id'] = globals().get('current_bar_id')
                except Exception:
                    pass
                return res
            except Exception as e:
                try:
                    log.exception('SafeOpen: execute_open_trade failed: %s', e)
                except Exception:
                    print('SafeOpen error:', e)
                return None
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
    return _safe_open

try:
    globals()['execute_open_trade'] = _make_safe_open()
except Exception:
    pass

# --- end minimal helpers ---

# --- Full improvements inserted by assistant ---
import threading as _threading
import functools as _functools

class OrderManager:
    """Central manager to serialize order openings and apply bar guards."""
    def __init__(self):
        try:
            import asyncio as _asyncio_local
            self._lock = _asyncio_local.Lock()
            self._asyncio = _asyncio_local
        except Exception:
            self._lock = None
            self._asyncio = None

    async def open_async(self, *args, **kwargs):
        try:
            if self._lock is not None:
                await self._lock.acquire()
        except Exception:
            pass
        try:
            cb = globals().get('current_bar_id')
            lclosed = globals().get('last_closed_bar_id')
            lopened = globals().get('last_opened_bar_id')
            if lclosed is not None and cb is not None and cb == lclosed:
                try:
                    log.info('OrderManager: skip open because last close on same bar %s', cb)
                except Exception:
                    pass
                return None
            if lopened is not None and cb is not None and cb == lopened:
                try:
                    log.info('OrderManager: skip open because already opened on bar %s', cb)
                except Exception:
                    pass
                return None
            orig = globals().get('_orig_execute_open_trade') or globals().get('execute_open_trade')
            if orig is None:
                try:
                    log.warning('OrderManager: original execute_open_trade not found')
                except Exception:
                    pass
                return None
            try:
                res = orig(*args, **kwargs)
                if self._asyncio and self._asyncio.iscoroutine(res):
                    res = await res
                try:
                    globals()['last_opened_bar_id'] = globals().get('current_bar_id')
                except Exception:
                    pass
                return res
            except Exception as e:
                try:
                    log.exception('OrderManager: open failed: %s', e)
                except Exception:
                    print('OrderManager open failed:', e)
                return None
        finally:
            try:
                if self._lock is not None and self._lock.locked():
                    try:
                        self._lock.release()
                    except Exception:
                        pass
            except Exception:
                pass

    def open(self, *args, **kwargs):
        if self._asyncio and self._asyncio.get_event_loop().is_running():
            try:
                return _functools.partial(self.open_async, *args, **kwargs)()
            except Exception:
                return None
        else:
            result = {}
            def _runner():
                try:
                    import asyncio as _asyncio_run
                    loop = _asyncio_run.new_event_loop()
                    _asyncio_run.set_event_loop(loop)
                    res = loop.run_until_complete(self.open_async(*args, **kwargs))
                    result['res'] = res
                except Exception as e:
                    result['err'] = str(e)
            t = _threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(timeout=5)
            return result.get('res', None)

# preserve original
try:
    if '_orig_execute_open_trade' not in globals() and 'execute_open_trade' in globals():
        globals()['_orig_execute_open_trade'] = globals()['execute_open_trade']
except Exception:
    pass

_ORDER_MANAGER = OrderManager()

def execute_open_trade(sym, wallet, side, size, price, extra=None, dry=True):
    try:
        return _ORDER_MANAGER.open(sym, wallet, side, size, price, extra=extra, dry=dry)
    except Exception as e:
        try:
            log.exception('execute_open_trade wrapper failed: %s', e)
        except Exception:
            print('execute_open_trade wrapper failed:', e)
        try:
            orig = globals().get('_orig_execute_open_trade')
            if orig:
                return orig(sym, wallet, side, size, price, extra=extra, dry=dry)
        except Exception:
            pass
        return None

def safe_fetch_df_sync(symbol, timeframe, limit):
    try:
        return OHLCV.sync_fetch(symbol, timeframe, limit)
    except Exception as e:
        try:
            log.warning('safe_fetch_df_sync failed for %s: %s', symbol, e)
        except Exception:
            print('safe_fetch_df_sync failed for', symbol, e)
        return None

async def safe_fetch_df_async(symbol, timeframe, limit):
    try:
        return await OHLCV.async_fetch(symbol, timeframe, limit)
    except Exception as e:
        try:
            log.warning('safe_fetch_df_async failed for %s: %s', symbol, e)
        except Exception:
            print('safe_fetch_df_async failed for', symbol, e)
        return None

def schedule_retrain(df, wnd_min, model_name):
    try:
        import asyncio as _asyncio_local2, threading as _threading_local2
        async def _runner():
            try:
                res = train_autoencoder(df, wnd_min, model_name)
                if _asyncio_local2.iscoroutine(res):
                    await res
            except Exception as e:
                try:
                    log.exception('schedule_retrain failed: %s', e)
                except Exception:
                    print('schedule_retrain failed:', e)
        try:
            loop = _asyncio_local2.get_event_loop()
            if loop.is_running():
                _asyncio_local2.create_task(_runner())
                return
        except Exception:
            pass
        def _thread_runner():
            try:
                _asyncio_local2.run(_runner())
            except Exception as e:
                try:
                    log.exception('retrain thread failed: %s', e)
                except Exception:
                    print('retrain thread failed:', e)
        _threading_local2.Thread(target=_thread_runner, daemon=True).start()
    except Exception as e:
        try:
            log.exception('schedule_retrain outer failed: %s', e)
        except Exception:
            print('schedule_retrain outer failed:', e)

# --- End full improvements ---
import dash_bootstrap_components as dbc

import lightgbm as lgb

# --- Helpers added by assistant: sanitize annotations, compute total PnL, marker helpers ---
def sanitize_annotations(fig):
    # Remove any 'ay' inside font dicts and ensure annotations have a top-level 'ay' offset.
    try:
        anns = getattr(getattr(fig, 'layout', None), 'annotations', None)
        if not anns:
            return
        cleaned = []
        for a in list(anns):
            try:
                ad = dict(a)
            except Exception:
                ad = {}
                for k in a:
                    try:
                        ad[k] = a[k]
                    except Exception:
                        pass
            f = ad.get('font')
            if isinstance(f, dict) and 'ay' in f:
                f2 = dict(f)
                f2.pop('ay', None)
                ad['font'] = f2
            if 'ay' in ad:
                try:
                    val = ad.pop('ay')
                    try:
                        n = int(val)
                    except Exception:
                        n = -15
                    ad['ay'] = n
                except Exception:
                    ad.pop('ay', None)
            if 'ay' not in ad and 'y' in ad:
                ad['ay'] = -15
            cleaned.append(ad)
        fig.update_layout(annotations=cleaned)
    except Exception:
        pass

def should_plot_marker(tr):
    try:
        if str(tr.get('status','')).lower() in ('closed','closed_by_system'):
            return False
    except Exception:
        pass
    return not bool(tr.get('_marker_removed'))

def mark_trade_closed(tr, exit_price=None, exit_time=None, pnl=None, pnl_scaled=None):
    try:
        tr.setdefault('_closed_info', {})
        tr['_closed_info'].update({
            'exit_price': exit_price if exit_price is not None else tr.get('exit_price'),
            'exit_time': exit_time if exit_time is not None else tr.get('exit_time'),
            'pnl': pnl if pnl is not None else tr.get('pnl'),
            'pnl_scaled': pnl_scaled if pnl_scaled is not None else tr.get('pnl_scaled'),
        })
    except Exception:
        pass
    tr['_marker_removed'] = True

def compute_total_pnl_since_start(store, fallback_realized=0.0):
    try:
        agg_real = 0.0
        agg_unreal = 0.0
        if isinstance(store, dict) and store.get('wallets'):
            for w in store.get('wallets', {}).values():
                for tt in w.get('trades', []):
                    try:
                        if tt.get('pnl_scaled') is not None:
                            agg_real += float(tt.get('pnl_scaled') or 0.0)
                        else:
                            agg_unreal += float(tt.get('pnl') or 0.0)
                    except Exception:
                        continue
        else:
            agg_real = float(fallback_realized or 0.0)
        return agg_real + agg_unreal
    except Exception:
        return float(fallback_realized or 0.0)

def get_total_realized_from_db(exclude_symbols=None):
    """Sum pnl_scaled from sqlite trades table, optionally excluding symbols present in memory.
    Returns float (0.0 on error)."""
    try:
        if exclude_symbols is None:
            exclude_symbols = []
        con = _db
        if con is None:
            return 0.0
        cur = con.cursor()
        if exclude_symbols:
            # prepare placeholders
            placeholders = ",".join("?" for _ in exclude_symbols)
            q = f"SELECT SUM(pnl_scaled) FROM trades WHERE pnl_scaled IS NOT NULL AND symbol NOT IN ({placeholders})"
            cur.execute(q, tuple(exclude_symbols))
        else:
            cur.execute("SELECT SUM(pnl_scaled) FROM trades WHERE pnl_scaled IS NOT NULL")
        row = cur.fetchone()
        s = float(row[0]) if row and row[0] is not None else 0.0
        return s
    except Exception:
        return 0.0

def compute_aggregated_realized(store, fallback_realized=0.0):
    """Unified aggregator for Total Realized PnL across:
       - in-memory server wallets (active_state['wallets'])
       - client-side store wallets (store['wallets'])
       - persisted DB entries (trades table) for symbols not present in wallets
       This avoids double-counting by summing in-memory/store values first, then adding DB totals only for symbols not already represented.
    """
    try:
        agg = 0.0
        seen_symbols = set()

        # merge server wallets into store if available (non-destructive)
        try:
            if isinstance(store, dict):
                with state_lock:
                    server_wallets = active_state.get('wallets', {}) or {}
                    if server_wallets:
                        store.setdefault('wallets', {})
                        for s, w in server_wallets.items():
                            # only copy shallow structure; assume wallet['trades'] contains pnl_scaled
                            store['wallets'][s] = w
        except Exception:
            pass

        # sum pnl_scaled from store wallets
        if isinstance(store, dict) and store.get('wallets'):
            for sym, w in store.get('wallets', {}).items():
                seen_symbols.add(sym)
                for tt in w.get('trades', []):
                    try:
                        if tt is None:
                            continue
                        # prefer pnl_scaled, fallback to pnl if necessary
                        if tt.get('pnl_scaled') is not None:
                            agg += float(tt.get('pnl_scaled') or 0.0)
                        elif tt.get('pnl') is not None:
                            # scale pnl to base_usd using stored notional if available
                            not_usd = float(tt.get('notional_usd') or max(abs(tt.get('size',0.0)) * float(tt.get('price') or 1.0), 1.0))
                            agg += float(tt.get('pnl') or 0.0) * (settings.base_usd / max(not_usd, 1e-8))
                    except Exception:
                        continue

        # add DB totals for symbols not in seen_symbols to include historical sessions
        try:
            db_sum = get_total_realized_from_db(exclude_symbols=list(seen_symbols) if seen_symbols else None)
            # db_sum may include already counted trades if DB contains same trades as store (rare).
            # We rely on excluding symbols above to reduce double-count risk.
            agg += float(db_sum or 0.0)
        except Exception:
            try:
                agg += float(fallback_realized or 0.0)
            except Exception:
                pass

        return float(agg)
    except Exception:
        try:
            return float(fallback_realized or 0.0)
        except Exception:
            return 0.0

# --- end helpers ---


# ==================== 1. SETTINGS ====================
class Settings:
    def __init__(self):

        # initial symbol (still can be changed by selector)
        self.symbol               = 'BTC/USDT'
        self.timeframes           = ['1h', '5m']
        self.history_limit        = 3000
        self.chart_limit          = 80
        self.chart_refresh_ms     = 10000

        # volatility / feature windows
        self.vol_window           = 60
        self.adx_window           = 14      # standard ADX window
        self.wnd_min              = 30
        self.wnd_step             = 15

        # money / sizing
        self.base_usd             = 10000
        # account for expected slippage (0.1%)
        self.slippage_pct         = 0.001

        # stops / trailing
        # slightly tighter stop by default (1.5%)
        self.stop_loss_pct        = 0.015
        # trailing should be wide enough for 5m/1h timeframes
        self.trailing_pct         = 0.001
        # ATR-based multipliers
        self.atr_multiplier_open  = 1.0   # use 1.0 as a conservative base
        self.stop_loss_atr_mult   = 1.5
        self.trailing_atr_mult    = 1.0
        self.max_trade_minutes    = 6 * 60

        # minimum spacing between trades (seconds) to avoid overtrading
        self.min_time_between_trades = 300

        # AE / training / architecture
        self.latent_dim           = 32
        self.k_neighbors          = 8
        # less frequent live training to reduce resource use
        self.live_train_interval  = 300
        self.dash_train_interval  = 600
        self.er_hist_size         = 1500
        self.open_softness_factor = 0.5
        self.exit_softness_factor = 1.2
        self.exit_prob_thr        = 0.85
        # target volatility (adjust if you observe over/under trading)
        self.target_vol           = 0.00020
        self.ae_lr                = 0.0002
        self.log_level            = 'INFO'

        # adaptivity: make system more permissive in low vol by default
        # baseline threshold nudged slightly higher to reduce false opens
        self.score_threshold      = 0.55
        self.min_ae_confidence    = 0.05
        self.use_cosine_latent    = True

        # Score & weights (can be changed live) - emphasize expected return (ER)
        self.w_breakout = 0.25
        self.w_er       = 0.45
        self.w_mom      = 0.20
        self.w_vol      = 0.10

        # Liquidity / market-cap settings
        self.liq_depth_pct       = 0.005
        self.liq_hist_size       = 500
        self.liq_min_usd         = 5000
        self.liq_max_usd         = 200000
        self.marketcap_weight    = 0.4
        self.liquidity_weight    = 0.6
        self.coingecko_enabled   = True
        self.coingecko_ttl       = 300

        # Selector / scanning - enforce Binance + USDT only, and safer defaults
        self.selector_interval     = 60 * 5
        self.selector_top_k        = 8
        self.selector_enabled      = True
        self.selector_orderbook_depth = 200
        self.selector_concurrency  = 6
        self.selector_min_liq_usd  = 10000
        self.selector_switch_margin = 0.05
        self.selector_max_idle_minutes = 5

        # scanning constraints (new flags)
        self.scan_binance_only     = True
        self.scan_usdt_only        = True
        self.skip_leveraged_tokens = True  # skip BULL/BEAR tokens

        # Dry-run (no real orders when True)
        self.dry_run = True

        # minimal profit required before trailing activates (fraction)
        self.min_profit_to_trail_pct = 0.001
        # minimum seconds between full retrains (cooldown)
        self.retrain_cooldown = 3600

settings = Settings()

# --- User-request: exclude specified base coins from selector/scans ---
EXCLUDED_BASES = {'WAVES'}  # never include these base coins in market scans (case-insensitive)

# ==================== 2. LOGGING ====================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'d': {'format': '%(asctime)s %(levelname)-8s %(message)s'}},
    'handlers': {'c': {'class': 'logging.StreamHandler','formatter':'d','level':settings.log_level}},
    'root': {'handlers': ['c'], 'level': settings.log_level}
}
dictConfig(LOGGING_CONFIG)
# --- Assistant helper: EV-based decision & open wrapper ---
import math as _math
def _simple_calibrate(score):
    try:
        s = float(score)
        return 1.0 / (1.0 + _math.exp(-s))
    except Exception:
        return 0.5

def compute_ev_and_decide_local(sym, wallet, side, proposed_size, price, score, df=None):
    try:
        p_win = _simple_calibrate(score)
        avg_win = 1.0
        avg_loss = 1.0
        try:
            trades = list(wallet.get('trades') or [])
            wins = []; losses = []
            for tr in reversed(trades):
                if tr.get('exit_price') is None:
                    continue
                try:
                    pnl = float(tr.get('pnl', tr.get('pnl_scaled', 0.0)))
                    if pnl >= 0: wins.append(pnl)
                    else: losses.append(abs(pnl))
                except Exception:
                    continue
                if len(wins)+len(losses) >= 100:
                    break
            if wins: avg_win = sum(wins)/len(wins)
            if losses: avg_loss = sum(losses)/len(losses)
            if avg_win <= 0: avg_win = 1.0
            if avg_loss <= 0: avg_loss = avg_win
        except Exception:
            pass
        try:
            fee_bps = getattr(settings, 'taker_fee_bps', None)
            if fee_bps is None:
                fee_bps = getattr(settings, 'taker_fee', 0.0007)
            fee_frac = fee_bps/10000.0 if fee_bps>1 else fee_bps
            spread_bps = getattr(settings, 'est_spread_bps', 0.0005)
            slippage = getattr(settings, 'est_slippage_bps', max(0.0002, spread_bps/2))
            total_bps = fee_frac + slippage
            costs = price * float(proposed_size) * total_bps
        except Exception:
            costs = 0.0
        ev = p_win * avg_win - (1-p_win) * avg_loss - costs
        final_size = proposed_size
        return (ev>0, final_size, 'ev_positive' if ev>0 else 'ev_negative', p_win, ev)
    except Exception:
        return (True, proposed_size, 'error', 0.5, 0.0)

def decide_and_open(sym, wallet, side, amt, price, score, df=None, extra=None, dry=True):
    try:
        ok, size, reason, p_win, ev = compute_ev_and_decide_local(sym, wallet, side, amt, price, score, df=df)
        try:
            log.debug("DecideOpen %s side=%s price=%.6f score=%.4f p=%.3f ev=%.4f reason=%s size=%s", sym, side, price, score, p_win, ev, reason, size)
        except Exception:
            pass
        if not ok:
            try:
                log.info("Open blocked by EV for %s side=%s p=%.3f ev=%.4f", sym, side, p_win, ev)
            except Exception:
                pass
            return None
        return execute_open_trade(sym, wallet, side, size, price, extra=extra, dry=dry)
    except Exception:
        try:
            log.exception("decide_and_open failed for %s", sym)
        except Exception:
            pass
        return execute_open_trade(sym, wallet, side, amt, price, extra=extra, dry=dry)

# --- end helper ---

log = logging.getLogger()
log.setLevel(settings.log_level)
log.info(f"AE initial lr: {settings.ae_lr}")

# ==================== 3. EXCHANGES ====================
async_ex = ccxt_async.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
sync_ex  = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

# ==================== 3b. Active state & locks ====================
active_state = {
    'active_symbol': settings.symbol,
    'wallets': {},          # per-symbol wallets: symbol -> {position, trades}
    'selector_task': None,
    'switch_lock': None
}
state_lock = threading.Lock()  # for synchronous threads (Dash callbacks) access safety

# ==================== 3c. SQLite persistence ====================
_db_path = "trades.db"

def init_db():
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        side TEXT,
        size REAL,
        entry_price REAL,
        exit_price REAL,
        pnl_scaled REAL,
        open_time TEXT,
        close_time TEXT,
        meta JSON
      )
    """)
    conn.commit()
    return conn

_db = init_db()

def save_trade_to_db(trade):
    try:
        cur = _db.cursor()
        cur.execute("""
          INSERT INTO trades (symbol, side, size, entry_price, exit_price, pnl_scaled, open_time, close_time, meta)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get('symbol'),
            trade.get('side'),
            float(trade.get('size') or 0.0),
            float(trade.get('price') or 0.0),
            float(trade.get('exit_price') or 0.0),
            float(trade.get('pnl_scaled') or 0.0),
            str(trade.get('time') or ''),
            str(trade.get('exit_time') or ''),
            json.dumps(trade, default=str)
        ))
        _db.commit()
    except Exception as e:
        log.warning(f"DB save failed: {e}")

def load_trades_from_db(symbol):
    """Load persisted trades for symbol from sqlite and return a wallet-like dict
    {'position':0.0,'trades':[...]} where trades have parsed timestamps (if available)."""
    try:
        cur = _db.cursor()
        cur.execute("""SELECT entry_price, exit_price, pnl_scaled, open_time, close_time, meta FROM trades WHERE symbol=? ORDER BY id ASC""", (symbol,))
        rows = cur.fetchall()
        trades = []
        for entry_price, exit_price, pnl_scaled, open_time, close_time, meta in rows:
            try:
                rec = json.loads(meta) if isinstance(meta, str) and meta else {}
                rec = {'_meta': rec}
            except Exception:
                rec = {}
            # coerce non-dict meta into dict to avoid .get on str
            if not isinstance(rec, dict):
                rec = {'_meta': rec}
            # map known columns into rec if missing
            if entry_price and 'price' not in rec:
                try:
                    rec['price'] = float(entry_price)
                except Exception:
                    rec['price'] = entry_price
            if exit_price is not None and rec.get('exit_price') is None:
                try:
                    rec['exit_price'] = float(exit_price)
                except Exception:
                    rec['exit_price'] = exit_price
            if pnl_scaled is not None and rec.get('pnl_scaled') is None:
                try:
                    rec['pnl_scaled'] = float(pnl_scaled)
                except Exception:
                    rec['pnl_scaled'] = pnl_scaled
            if open_time and rec.get('time') is None:
                try:
                    rec['time'] = pd.to_datetime(open_time, utc=True).tz_convert('Europe/Kyiv')
                except Exception:
                    rec['time'] = open_time
            if close_time and rec.get('exit_time') is None:
                try:
                    rec['exit_time'] = pd.to_datetime(close_time, utc=True).tz_convert('Europe/Kyiv')
                except Exception:
                    rec['exit_time'] = close_time
            trades.append(rec)
        return {'position': 0.0, 'trades': trades}
    except Exception as e:
        log.warning(f"load_trades_from_db failed: {e}")
        return {'position': 0.0, 'trades': []}

# ==================== 4. DATA LOADER ====================
class OHLCV:
    @staticmethod
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def async_fetch(symbol, timeframe, limit):
        raw = await async_ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
        df['ts'] = pd.to_datetime(df['ts'],unit='ms',utc=True).dt.tz_convert('Europe/Kyiv')
        return df

    @staticmethod
    def sync_fetch(symbol, timeframe, limit):
        raw = sync_ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
        df['ts'] = pd.to_datetime(df['ts'],unit='ms',utc=True).dt.tz_convert('Europe/Kyiv')
        return df

# ==================== 5. UTILITY ====================

def execute_open_trade(sym, wallet, side, size, price, extra=None, dry=True):
    """Synchronous helper that records an open trade in the in-memory wallet and (optionally)
    schedules a background async order placement when dry=False. Returns the trade dict.
    This is intentionally conservative and will not block the event loop when placing real orders.
    """
    try:
        now_ts = time.time()
        now_iso = datetime.datetime.now(tz=datetime.timezone.utc).astimezone().isoformat()
        tr = {
            'symbol': sym,
            'side': side,
            'type': side,
            'size': float(size),
            'price': float(price),
            'entry_price': float(price),
            'open_time': now_iso,
            'time_ts': now_ts,
            'open_idx': None,
            'notional_usd': float(size * price),
            'meta': dict(extra) if isinstance(extra, dict) else {},
            'executed': False,
            'order_info': None,
        }
        if extra and isinstance(extra, dict):
            # copy commonly used extras
            for k in ('open_idx','atr','er_raw','ae_conf','notional_usd'):
                if k in extra:
                    tr[k] = extra[k]
                    tr['meta'][k] = extra[k]

        # ensure wallet structure
        if 'trades' not in wallet:
            wallet['trades'] = []
        wallet['trades'].append(tr)

        # set position sign convention: Long positive, Short negative
        if side.lower().startswith('long'):
            wallet['position'] = float(size)
            try:
                wallet['position_changed_ts'] = time.time()
            except Exception:
                pass
        else:
            wallet['position'] = -float(size)
            try:
                wallet['position_changed_ts'] = time.time()
            except Exception:
                pass

        # persist quickly if DB helper exists
        try:
            save_trade_to_db({
                'symbol': tr['symbol'],
                'side': tr['side'],
                'size': tr['size'],
                'entry_price': tr['entry_price'],
                'exit_price': None,
                'pnl_scaled': None,
                'open_time': tr['open_time'],
                'close_time': None,
                'meta': json.dumps(tr.get('meta', {}))
            })
        except Exception:
            # non-fatal if DB not available
            pass

        # If not dry, attempt to place real order asynchronously to avoid blocking
        if not dry:
            async def _place():
                try:
                    # prefer async_ex.create_order / create_market_buy/sell depending on exchange wrapper
                    if hasattr(async_ex, 'create_order') and asyncio.iscoroutinefunction(async_ex.create_order):
                        side_map = {'Long':'buy','Short':'sell'}
                        order_side = side_map.get(side, 'buy')
                        # place market order for symbol
                        try:
                            ord_res = await async_ex.create_order(sym, 'market', order_side, size)
                        except TypeError:
                            # some wrappers require price param for limit orders; fall back to market order signature
                            ord_res = await async_ex.create_order(sym, 'market', order_side, size, None)
                        tr['executed'] = True
                        tr['order_info'] = ord_res
                    else:
                        # try common convenience methods
                        placed = False
                        if hasattr(async_ex, 'market_buy') and hasattr(async_ex, 'market_sell'):
                            if side.lower().startswith('long'):
                                ord_res = await async_ex.market_buy(sym, size)
                            else:
                                ord_res = await async_ex.market_sell(sym, size)
                            tr['executed'] = True
                            tr['order_info'] = ord_res
                except Exception as e:
                    log.exception("Async place order failed: %s", e)
            try:
                asyncio.create_task(_place())
            except Exception:
                pass

        return tr
    except Exception as e:
        log.exception("execute_open_trade failed: %s", e)
        return None

def execute_close_trade(sym, wallet, tr, exit_price, df_for_feat=None, reason=None, dry=True):
    """Close trade helper — updates trade dict, wallet and persists to DB; returns pnl_scaled (or None).
    Ensures Telegram message is sent and DB entry created. """
    # DIAG: log entry to execute_close_trade
    try:
        log.debug(f"[DIAG] execute_close_trade called: sym={sym}, exit_price={exit_price}, tr_id={tr.get('id')}, dry={dry}")
    except Exception:
        pass

    try:
        # Idempotency guard: skip if already closed
        if tr.get('pnl_scaled') is not None or tr.get('exit_price') is not None or tr.get('closed_ts') is not None:
            log.info(f"execute_close_trade: trade already closed (sym={sym}, id={tr.get('id','')}) - skipping")
            return tr.get('pnl_scaled')

        # Normalize times
        exit_ts = time.time()
        try:
            exit_iso = pd.Timestamp.now(tz='Europe/Kyiv').isoformat()
        except Exception:
            exit_iso = datetime.datetime.now(tz=datetime.timezone.utc).astimezone().isoformat()

        # update tr fields
        tr['exit_price'] = float(exit_price)
        tr['exit_time'] = exit_iso
        # attempt to compute an aligned candle timestamp (ISO) if caller provided df_for_feat
        try:
            if df_for_feat is not None and hasattr(df_for_feat, 'index') and len(df_for_feat.index) > 0:
                exit_dt = pd.to_datetime(exit_iso)
                try:
                    idx = df_for_feat.index.get_indexer([exit_dt], method='nearest')[0]
                    tr['exit_time_aligned'] = pd.to_datetime(df_for_feat.index[idx]).isoformat()
                except Exception:
                    tr['exit_time_aligned'] = exit_iso
        except Exception:
            tr['exit_time_aligned'] = exit_iso
        # compute pnl: Long positive, Short negative
        typ = tr.get('type') or tr.get('side') or 'Long'
        entry_price = float(tr.get('price') or tr.get('entry_price') or exit_price)
        size_amt = float(tr.get('size') or 0.0)
        if str(typ).lower().startswith('long'):
            pnl_usdt = (tr['exit_price'] - entry_price) * size_amt
        else:
            pnl_usdt = (entry_price - tr['exit_price']) * size_amt
        tr['pnl'] = float(pnl_usdt)
        not_usd = float(tr.get('notional_usd') or max(abs(size_amt) * entry_price, 1.0))
        tr['pnl_scaled'] = float(pnl_usdt * (settings.base_usd / max(not_usd, 1e-8)))

        # metadata
        tr['closed_reason'] = reason
        tr['closed_ts'] = exit_ts

        # finalize wallet
        try:
            wallet['position'] = 0.0
        except Exception:
            wallet['position'] = 0.0
        try:
            wallet['position_changed_ts'] = time.time()
        except Exception:
            pass

        # persist to DB (best-effort)
        try:
            save_trade_to_db({
                'symbol': tr.get('symbol') or sym,
                'side': tr.get('side') or tr.get('type'),
                'size': tr.get('size'),
                'price': tr.get('price') or tr.get('entry_price'),
                'exit_price': tr.get('exit_price'),
                'pnl_scaled': tr.get('pnl_scaled'),
                'time': tr.get('time') or tr.get('open_time'),
                'exit_time': tr.get('exit_time_aligned') or tr.get('exit_time'),
                'meta': tr
            })
        except Exception as e:
            log.warning(f"execute_close_trade: DB save failed: {e}")

        # send Telegram exit message if formatting helper exists
        try:
            ot = tr.get('time') or tr.get('open_time')
            open_time_parsed = ot
            try:
                open_time_parsed = pd.to_datetime(ot)
            except Exception:
                pass
            try:
                msg = format_exit_signal(sym, tr.get('type') or tr.get('side',''), float(tr.get('price') or entry_price), float(tr.get('exit_price')), float(tr.get('pnl_scaled') or 0.0), reason, open_time_parsed)
                send_telegram(msg)
            except Exception:
                # fallback simple text
                try:
                    send_telegram(f"EXIT {sym} {tr.get('type')} pnl={tr.get('pnl'):+.2f} reason={reason}")
                except Exception:
                    pass
        except Exception:
            pass
        return tr.get('pnl_scaled')
    except Exception as e:
        log.exception(f"execute_close_trade failed: {e}")
        # Best-effort fallback: set wallet position to 0
        try:
            wallet['position'] = 0.0
        except Exception:
            pass
        return None


def compute_volatility(close, window):
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return 0.0
    if len(close) < window + 1:
        returns = np.diff(np.log(close + 1e-8))
    else:
        returns = np.diff(np.log(close[-(window+1):] + 1e-8))
    return float(np.std(returns))

def compute_atr(df, window=14):
    high, low, close = df['h'], df['l'], df['c']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    val = tr.rolling(window).mean().iloc[-1]
    if np.isnan(val) or val <= 0:
        return float((high - low).iloc[-1] if len(df) >= 1 else 1e-6)
    return float(val)

def safe_norm(value, lo, hi, eps=1e-8):
    denom = hi - lo
    if denom <= eps:
        if value > hi:
            return 1.0
        if value < lo:
            return 0.0
        return 0.5
    return float(np.clip((value - lo) / denom, 0.0, 1.0))

# ==================== 5b. Liquidity & MarketCap helpers ====================
_coingecko_cache = {'ts': 0.0, 'symbol': None, 'market_cap': None}

def fetch_marketcap_coingecko(symbol):
    if not getattr(settings, 'coingecko_enabled', True):
        return None
    try:
        now = time.time()
        base = symbol.split('/')[0].lower()
        if (_coingecko_cache.get('symbol') == base) and (now - _coingecko_cache.get('ts',0) < settings.coingecko_ttl):
            return _coingecko_cache.get('market_cap')

        cg_map = {
            'btc': 'bitcoin', 'eth': 'ethereum', 'xrp': 'ripple', 'ada':'cardano', 'bnb':'binancecoin',
            'sol':'solana', 'doge':'dogecoin', 'ltc':'litecoin', 'link':'chainlink', 'matic':'polygon'
        }
        cg_id = cg_map.get(base, base)
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {'vs_currency':'usd', 'ids': cg_id}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data)>0 and 'market_cap' in data[0]:
                mc = float(data[0].get('market_cap') or 0.0)
                _coingecko_cache.update({'ts': now, 'symbol': base, 'market_cap': mc})
                return mc
    except Exception as e:
        log.debug(f"CoinGecko marketcap fetch failed for {symbol}: {e}")
    return None

liq_state = {'hist': collections.deque(maxlen=settings.liq_hist_size)}

def compute_liquidity_score_from_orderbook(order_book, price, depth_pct=None):
    if depth_pct is None:
        depth_pct = getattr(settings, 'liq_depth_pct', 0.005)
    try:
        mid = float(price)
        low = mid * (1 - depth_pct)
        high = mid * (1 + depth_pct)
        bids = order_book.get('bids', [])[:500]
        asks = order_book.get('asks', [])[:500]
        depth_usd = 0.0
        for p, amt in bids:
            p_f = float(p); a_f = float(amt)
            if p_f >= low:
                depth_usd += p_f * a_f
        for p, amt in asks:
            p_f = float(p); a_f = float(amt)
            if p_f <= high:
                depth_usd += p_f * a_f
        try:
            liq_state['hist'].append(float(depth_usd))
        except Exception:
            pass
        lo = getattr(settings, 'liq_min_usd', 1000)
        hi = getattr(settings, 'liq_max_usd', 200000)
        if hi <= lo:
            hi = lo * 10.0
        score = safe_norm(depth_usd, lo, hi)
        return float(depth_usd), float(score)
    except Exception as e:
        log.debug(f"compute_liquidity_score error: {e}")
        return 0.0, 0.0

# ==================== 6. AUTOENCODER & PATTERN MATCHER ====================
class WinDS(Dataset):
    def __init__(self, arr, m):
        wins = np.lib.stride_tricks.sliding_window_view(arr,(m,2))
        self.windows = wins.reshape(-1,m,2).astype(np.float32)
    def __len__(self): return len(self.windows)
    def __getitem__(self,idx):
        x = self.windows[idx]
        noise = torch.randn_like(torch.from_numpy(x)) * 0.01
        return torch.from_numpy(x) + noise, torch.from_numpy(x)

def make_online_dataloader(df, m, batch_size=64):
    arr = df[['c','v']].values.astype(np.float32)
    ds  = WinDS(arr,m) if len(arr) >= m else WinDS(np.zeros((m,2),dtype=np.float32).reshape(1,m,2), m)
    N   = len(ds)
    if N == 0:
        arr_small = np.zeros((1,m,2), dtype=np.float32)
        ds = WinDS(arr_small, m)
        N = len(ds)
    k   = int(max(1, N * 0.2))
    ds2 = ConcatDataset([
        Subset(ds, list(range(max(0,N-k), N))),
        Subset(ds, list(range(min(N, N-k))))
    ])
    return DataLoader(ds2, batch_size=min(batch_size, len(ds2)), shuffle=True)

class ConvAutoEncoder(nn.Module):
    def __init__(self, m, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2,16,3,padding=1), nn.ReLU(),
            nn.Conv1d(16,8,3,padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(8*m, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 8*m),
            nn.Unflatten(1, (8,m)),
            nn.ConvTranspose1d(8,16,3,padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,2,3,padding=1)
        )
    def forward(self, x):
        x = x.permute(0,2,1)
        z = self.enc(x)
        out = self.dec(z)
        return out.permute(0,2,1), z

ae_lr_map = {}
def train_autoencoder(df, m, fp, epochs=10, base_lr=None):
    lr = ae_lr_map.get(m, base_lr or settings.ae_lr)
    dl = make_online_dataloader(df, m)
    model = ConvAutoEncoder(m, settings.latent_dim).train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        tot = 0.0
        total_samples = 0
        for x, y in dl:
            recon, _ = model(x)
            loss = loss_fn(recon, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
            total_samples += x.size(0)
        avg = tot / max(1, total_samples)
        log.info(f"[ConvAE m={m}] Ep {ep}/{epochs} — Loss {avg:.6f} lr={lr:.6g}")
        lr *= 0.99
        for pg in opt.param_groups:
            pg['lr'] = lr
    try:
        torch.save(model.state_dict(), fp)
        ae_lr_map[m] = lr
        log.info(f"Saved ConvAE m={m} → {fp} lr={lr:.6g}")
    except Exception as e:
        log.warning(f"Saving AE failed: {e}")
    return model

class PatternMatcherAE:
    def __init__(self, df, m):
        self.m = m
        arr = df[['c','v']].values.astype(np.float32)
        self.fp = f"conv_ae_m{m}.pth"
        if not os.path.isfile(self.fp):
            train_autoencoder(df, m, self.fp)

        ae = ConvAutoEncoder(m, settings.latent_dim)
        try:
            ae.load_state_dict(torch.load(self.fp, map_location='cpu'))
        except Exception:
            train_autoencoder(df, m, self.fp)
            ae.load_state_dict(torch.load(self.fp, map_location='cpu'))
        ae.eval()
        self.ae = ae

        wins = np.lib.stride_tricks.sliding_window_view(arr, (m, 2)).reshape(-1, m, 2).copy()
        prices = wins[..., 0]
        vols   = wins[..., 1]
        mean_p = prices.mean(axis=1, keepdims=True)
        std_p  = prices.std(axis=1, keepdims=True) + 1e-8
        norm_p = (prices - mean_p) / std_p
        logv   = np.log1p(vols)
        mean_v = logv.mean(axis=1, keepdims=True)
        std_v  = logv.std(axis=1, keepdims=True) + 1e-8
        norm_v = (logv - mean_v) / std_v
        norm_wins = np.stack([norm_p, norm_v], axis=2).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(norm_wins).float()
            _, Z = ae(x)
        Z = Z.numpy()

        if settings.use_cosine_latent and _has_faiss:
            norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
            Zn = Z / norms
            self.index = faiss.IndexFlatIP(Zn.shape[1])
            self.index.add(Zn)
            self.Z = Zn
            self.cosine = True
        else:
            # fallback to L2 index or plain storage
            if _has_faiss:
                self.index = faiss.IndexFlatL2(Z.shape[1])
                self.index.add(Z)
            else:
                # fallback: store Z and do brute force search
                self.index = None
            self.Z = Z
            self.cosine = False

        self.df = df.reset_index(drop=True)

    def match(self):
        arr = self.df[['c','v']].values.astype(np.float32)
        if len(arr) < self.m + 1:
            prices = arr[:, 0]
            fallback = float(prices[-1] / prices[-(settings.adx_window + 1)] - 1) if len(prices) > settings.adx_window else 0.0
            return fallback, 0.05

        cur = arr[-self.m:].reshape(1, self.m, 2).copy()
        prices = cur[..., 0]
        vols   = cur[..., 1]
        mean_p = prices.mean(axis=1, keepdims=True)
        std_p  = prices.std(axis=1, keepdims=True) + 1e-8
        norm_p = (prices - mean_p) / std_p
        logv   = np.log1p(vols)
        mean_v = logv.mean(axis=1, keepdims=True)
        std_v  = logv.std(axis=1, keepdims=True) + 1e-8
        norm_v = (logv - mean_v) / std_v
        cur_norm = np.stack([norm_p, norm_v], axis=2).astype(np.float32)

        with torch.no_grad():
            _, zc = self.ae(torch.from_numpy(cur_norm).float())
        zc = zc.numpy()

        if self.cosine and _has_faiss and self.index is not None:
            zcn = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + 1e-8)
            D, I = self.index.search(zcn, settings.k_neighbors)
            sims = D[0]
            weights = np.exp(sims / (sims.max()+1e-8))
        elif _has_faiss and self.index is not None:
            D, I = self.index.search(zc, settings.k_neighbors)
            d = D[0]
            thr = np.median(d) + 1e-8
            weights = np.exp(-d / (thr + 1e-8))
        else:
            # brute-force fallback: L2 distances
            dists = np.linalg.norm(self.Z - zc, axis=1)
            idxs = np.argsort(dists)[:settings.k_neighbors]
            d = dists[idxs]
            thr = np.median(d) + 1e-8
            weights = np.exp(-d / (thr + 1e-8))
            I = np.array([idxs])

        rets, ws = [], []
        for wgt, idx in zip(weights, I[0]):
            fut = idx + self.m + settings.adx_window
            if fut < len(self.df):
                price_match  = self.df.loc[idx + self.m, 'c']
                price_future = self.df.loc[fut, 'c']
                rets.append(float(price_future / price_match - 1))
                ws.append(float(wgt))
        if rets:
            rets = np.array(rets)
            ws   = np.array(ws)
            wsum = ws.sum() + 1e-8
            er = float((rets * ws).sum() / wsum)
            conf = float(min(1.0, (wsum / settings.k_neighbors)))
            return er, conf

        prices = arr[:, 0]
        fallback = float(prices[-1] / prices[-(settings.adx_window + 1)] - 1) if len(prices) > settings.adx_window else 0.0
        return fallback, 0.05

# ==================== 7. EXIT MODEL ====================
exit_model        = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
X_history, y_history = [], []
retrain_interval  = 3600
last_retrain_ts   = None

def make_exit_feature(df, trade, er, ae_conf=0.0):
    m = min(settings.wnd_min, len(df)-1)
    sub = df[['c','v']].iloc[-m:] if m>0 else pd.DataFrame({'c':[0.0],'v':[0.0]})
    bars = (len(df)-1) - trade.get('open_idx', max(0,len(df)-1))
    cvals = sub.c.values if 'c' in sub else np.zeros(m)
    vvals = sub.v.values if 'v' in sub else np.zeros(m)
    f = np.concatenate([cvals, vvals, [er, ae_conf, bars]])
    return f.reshape(1, -1)

# ==================== 8. SMS FORMATTING & HELPERS ====================
def format_price_for_display(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    # For prices >= 1: show integer if whole, else two decimals
    if p >= 1:
        if abs(p - round(p)) < 1e-9:
            return f"{int(round(p))}"
        return f"{p:.2f}"
    # For prices < 1: show exactly 3 decimals (e.g., 0.000)
    return f"{p:.3f}"

def build_metrics_cards(metrics_list):
    # accepts a list with a single metrics dict or empty list
    m = metrics_list[0] if metrics_list and isinstance(metrics_list, list) else {}
    # safe-get helpers
    def g(k, default='-'):
        return m.get(k, default) if isinstance(m, dict) else default
    equity = g('equity', '-')
    realized = g('total_realized_pnl', '-')
    unreal = g('unrealized_pnl', '-')
    open_pos = g('open_position', 'Neutral')
    winrate = g('win_rate', '-')
    cards = [
        dbc.Card([dbc.CardBody([html.Div('Equity', style={'fontSize':'12px','color':'#a9cbdc'}), html.Div(str(equity), id='card-equity', style={'fontSize':'20px','fontWeight':'700'})])], style={'flex':'1','backgroundColor':'#071b2a','border':'1px solid rgba(255,255,255,0.03)'}),
        dbc.Card([dbc.CardBody([html.Div('Total Realized PnL', style={'fontSize':'12px','color':'#a9cbdc'}), html.Div(str(realized), id='card-realized', style={'fontSize':'18px','fontWeight':'700'})])], style={'width':'220px','backgroundColor':'#052e20'}),
        dbc.Card([dbc.CardBody([html.Div('Open Position', style={'fontSize':'12px','color':'#a9cbdc'}), html.Div(str(open_pos), id='card-open-pos', style={'fontSize':'16px','fontWeight':'700'})])], style={'width':'220px','backgroundColor':'#10202a'}),
        dbc.Card([dbc.CardBody([html.Div('Unrealized PnL (current)', style={'fontSize':'12px','color':'#a9cbdc'}), html.Div(str(unreal), id='card-unrealized', style={'fontSize':'16px','fontWeight':'700'})])], style={'width':'220px','backgroundColor':'#071225'}),
        dbc.Card([dbc.CardBody([html.Div('Win Rate', style={'fontSize':'12px','color':'#a9cbdc'}), html.Div(str(winrate), id='card-winrate', style={'fontSize':'16px','fontWeight':'700'})])], style={'width':'140px','backgroundColor':'#071225'}),
    ]

    # --- Replace Total Realized PnL card with enhanced visual version ---
    try:
        try:
            _realized_val = float(realized) if isinstance(realized, (int,float)) else float(str(realized).replace(',','')) if realized not in (None,'-') else 0.0
        except Exception:
            _realized_val = 0.0

        if _realized_val > 0:
            _arrow = '▲'
            _arrow_color = '#0f9d58'
        elif _realized_val < 0:
            _arrow = '▼'
            _arrow_color = '#db4437'
        else:
            _arrow = '⏺'
            _arrow_color = '#9aa0a6'

        _db_included = False
        try:
            db_sum = get_total_realized_from_db()
            if abs(float(db_sum or 0.0)) > 1e-6:
                _db_included = True
        except Exception:
            _db_included = False

        _note = 'Includes historical (DB)' if _db_included else 'Live session only'

        new_realized_card = dbc.Card([dbc.CardBody([
            html.Div('Total Realized PnL', style={'fontSize':'12px','color':'#cfe8d1','opacity':0.9}),
            html.Div([
                html.Span(_arrow, id='card-realized-arrow', style={'fontSize':'16px','paddingRight':'6px','color':_arrow_color}),
                html.Span(str(realized), id='card-realized', style={'fontSize':'20px','fontWeight':'700','letterSpacing':'0.4px','color':_arrow_color})
            ], style={'display':'flex','alignItems':'center'}),
            html.Div(_note, id='card-realized-note', style={'fontSize':'11px','opacity':0.8,'marginTop':'6px'})
        ])], style={'width':'220px','backgroundColor':'#052e20','borderRadius':'8px','boxShadow':'0 2px 6px rgba(0,0,0,0.35)'})
        # second card in original list corresponds to realized; replace safely if index exists
        if isinstance(cards, list) and len(cards) > 1:
            cards[1] = new_realized_card
    except Exception:
        # keep original in case of any error
        pass

    # --- Small legend/help card for visual cues (added by assistant) ---
    try:
        legend_card = dbc.Card([dbc.CardBody([
            html.Div('Legend', style={'fontSize':'12px','fontWeight':'700','marginBottom':'6px'}),
            html.Div([html.Span('▲ Positive realized', style={'color':'#0f9d58','marginRight':'12px'}), html.Span('▼ Negative realized', style={'color':'#db4437','marginRight':'12px'}), html.Span('⏺ Neutral', style={'color':'#9aa0a6'})], style={'fontSize':'11px','opacity':0.9})
        ])], style={'width':'220px','backgroundColor':'transparent','border':'1px dashed rgba(255,255,255,0.03)','borderRadius':'8px'})
        # append legend if not already appended
        if isinstance(cards, list):
            cards.append(legend_card)
    except Exception:
        pass

    return cards

def format_open_signal(symbol, side, price, size, er, vol, bu, spike):
    now       = pd.Timestamp.now(tz='Europe/Kyiv').strftime('%H:%M')
    direction = '📈 Up' if bu else '📉 Down'
    spike_ic  = '📊 Yes' if spike else '📊 No'
    strength  = min(abs(er) / 0.05, 1.0) if er is not None else 0.0
    rr        = (er / settings.stop_loss_pct) if getattr(settings,'stop_loss_pct',0) > 0 and er is not None else float('inf')
    notional  = int(round(size * price)) if size is not None else 0
    # Compact HTML message (size intentionally omitted to reduce noise)
    return (
        f"⏰ <b>{now}</b> | <b>🚀 {side} OPEN</b>\n"
        f"<b>{symbol}</b> • <code>{price:.6f}</code>\n"
        f"Notional: <code>{notional}</code> • Strength: <b>{strength:.2f}</b>\n"
        f"ER: <b>{(er if er is not None else 0):.4f}</b> • Vol: {vol*100:.2f}% • R:R: {rr:.2f}\n"
        f"Direction: {direction} • Spike: {spike_ic}\n"
    )
def format_exit_signal(symbol, side, entry, exit_p, pnl_scaled, reason, open_time):
    if not isinstance(open_time, pd.Timestamp):
        try:
            open_time = pd.to_datetime(open_time, utc=True).tz_convert('Europe/Kyiv')
        except Exception:
            open_time = pd.Timestamp.now(tz='Europe/Kyiv')
    now = pd.Timestamp.now(tz=open_time.tzinfo)
    dur = now - open_time
    h, rem = divmod(int(dur.total_seconds()), 3600)
    m = rem // 60
    reason_text = {
        'trailing':  '⏱ Trailing-stop',
        'model_exit':'🤖 Model-exit',
        'stop': '🛑 Stop-loss',
        'timeout':'⌛ Time-exit'
    }.get(reason, reason)
    try:
        pct = (pnl_scaled / settings.base_usd * 100.0) if settings.base_usd else 0.0
    except Exception:
        pct = 0.0
    pnl_int = int(round(pnl_scaled))
    # Richer HTML-formatted exit message
    return (
        f"⏰ <b>{now.strftime('%H:%M')}</b> | <b>🔒 EXIT {side}</b>\n"
        f"<b>{symbol}</b> • Entry: <code>{entry:.6f}</code> → Exit: <code>{exit_p:.6f}</code>\n"
        f"<b>PnL:</b> <code>{pnl_int:+d}</code> ({pct:+.2f}% of {int(settings.base_usd)})\n"
        f"Duration: <b>{h}h {m}m</b> • {reason_text}\n"
    )

def ensure_trade_timestamps(trades):
    """
    Преобразує рядкові timestamp (ISO) назад у pd.Timestamp з таймзоною Europe/Kyiv.
    Мутує список trades на місці.
    """
    if not trades:
        return
    for t in trades:
        # time
        try:
            tt = t.get('time')
            if isinstance(tt, str):
                # parse as UTC then convert
                t['time'] = pd.to_datetime(tt, utc=True).tz_convert('Europe/Kyiv')
            elif tt is None:
                t['time'] = pd.Timestamp.now(tz='Europe/Kyiv')
            # if it's already a Timestamp, keep it
        except Exception:
            try:
                t['time'] = pd.to_datetime(t.get('time'))
            except Exception:
                t['time'] = pd.Timestamp.now(tz='Europe/Kyiv')

        # exit_time (may be None)
        try:
            et = t.get('exit_time')
            if isinstance(et, str):
                t['exit_time'] = pd.to_datetime(et, utc=True).tz_convert('Europe/Kyiv')
            elif et is None:
                t['exit_time'] = None
        except Exception:
            t['exit_time'] = None

# ==================== Adaptive volatility state & function ====================
vol_state = {
    'ema': None,
    'alpha': 0.12,
    'hist': collections.deque(maxlen=1000)
}

def compute_adaptive_params(current_vol, liquidity_score=0.0, market_cap=None):
    vol_state['hist'].append(float(current_vol))
    if vol_state['ema'] is None:
        vol_state['ema'] = float(current_vol)
    else:
        a = vol_state['alpha']
        vol_state['ema'] = a * float(current_vol) + (1 - a) * vol_state['ema']

    vol_val = vol_state['ema']
    if len(vol_state['hist']) >= 20:
        lo = np.percentile(vol_state['hist'], 10)
        hi = np.percentile(vol_state['hist'], 90)
    else:
        lo = min(vol_state['hist']) if vol_state['hist'] else vol_val * 0.5
        hi = max(vol_state['hist']) if vol_state['hist'] else vol_val * 1.5

    vol_scale = safe_norm(vol_val, lo, hi)
    vol_scale = vol_scale ** 1.2

    try:
        if market_cap is None or market_cap <= 0:
            mc_score = 0.5
        else:
            mc_lo, mc_hi = 1e7, 1e12
            mc_score = safe_norm(np.log10(market_cap), np.log10(mc_lo), np.log10(mc_hi))
    except Exception:
        mc_score = 0.5

    lw = getattr(settings, 'liquidity_weight', 0.6)
    mw = getattr(settings, 'marketcap_weight', 0.4)
    combined_liq = float(np.clip((liquidity_score * lw + mc_score * mw), 0.0, 1.0))

    score_thr = float(np.clip(settings.score_threshold * (1.0 + 0.6 * vol_scale) * (1.0 - 0.35 * combined_liq), 0.05, 0.95))
    min_time = int(max(1, settings.min_time_between_trades * (1.0 + 4.0 * vol_scale)))
    atr_open = float(np.clip(settings.atr_multiplier_open * (1.0 + 1.0 * vol_scale), 0.01, 3.0))
    open_soft = float(np.clip(settings.open_softness_factor * (1.0 - 0.6 * vol_scale), 0.01, 1.0))
    min_ae = float(np.clip(settings.min_ae_confidence * (1.0 + 3.0 * vol_scale), 0.0, 0.6))
    kelly_cap = float(np.clip(0.1 * (1.0 - 0.7 * vol_scale) + 0.01 + 0.04 * combined_liq, 0.01, 0.25))
    stop_loss_atr = float(np.clip(settings.stop_loss_atr_mult * (1.0 - 0.4 * (1 - vol_scale)), 0.2, 3.0))

    return {
        'vol_scale': vol_scale,
        'score_threshold': score_thr,
        'min_time_between_trades': min_time,
        'atr_multiplier_open': atr_open,
        'open_softness_factor': open_soft,
        'min_ae_confidence': min_ae,
        'kelly_cap': kelly_cap,
        'stop_loss_atr_mult': stop_loss_atr,
        'vol_ema': vol_val,
        'vol_lo': lo,
        'vol_hi': hi,
        'liquidity_score': combined_liq,
        'marketcap_score': mc_score,
        'market_cap': market_cap,
        'liquidity_usd_recent': (liq_state['hist'][-1] if liq_state['hist'] else 0.0)
    }

# ==================== 9. Selector (market scanner) — покращений ====================
async def quick_score_for_symbol(sym, sem):
    """Швидкий метрик: orderbook liquidity + 24h volume + price momentum"""
    async with sem:
        try:
            tick = await async_ex.fetch_ticker(sym)
            price = float(tick.get('last') or tick.get('info',{}).get('lastPrice', 0.0))
            vol24 = float(tick.get('quoteVolume') or tick.get('baseVolume') or 0.0)
        except Exception:
            price = 0.0
            vol24 = 0.0

        liq_usd, liq_score = 0.0, 0.0
        try:
            ob = await async_ex.fetch_order_book(sym, limit=getattr(settings,'selector_orderbook_depth',200))
            liq_usd, liq_score = compute_liquidity_score_from_orderbook(ob, price)
        except Exception as e:
            log.debug(f"Selector OB failed {sym}: {e}")

        # simple momentum: last/6h (attempt fetch small ohlcv)
        mom = 0.0
        try:
            df = await OHLCV.async_fetch(sym, '1h', 50)
            if len(df)>1:
                mom = float(df['c'].iloc[-1] / df['c'].iloc[-6] - 1) if len(df)>6 else 0.0
        except Exception:
            mom = 0.0

        # normalized score combine
        score = 0.55 * liq_score + 0.25 * safe_norm(vol24, 1e3, 1e8) + 0.2 * safe_norm(abs(mom), 0, 0.1)
        return sym, score, {'liq_usd': liq_usd, 'liq_score': liq_score, 'vol24': vol24, 'mom': mom}

async def get_current_quick_score(sym):
    """Helper — quick score for current symbol (sequential)"""
    try:
        tick = await async_ex.fetch_ticker(sym)
        price = float(tick.get('last') or 0.0)
        ob = await async_ex.fetch_order_book(sym, limit=getattr(settings,'selector_orderbook_depth',200))
        liq_usd, liq_score = compute_liquidity_score_from_orderbook(ob, price)
        df = await OHLCV.async_fetch(sym, '1h', 24)
        mom = float(df['c'].iloc[-1] / df['c'].iloc[-6] - 1) if len(df)>6 else 0.0
        score = 0.55 * liq_score + 0.25 * safe_norm(0, 1e3, 1e8) + 0.2 * safe_norm(abs(mom), 0, 0.1)
        # NOTE: earlier this had a bug using 0 instead of vol24; left intentionally minimal here for quick check
        return score
    except Exception:
        return 0.0

# ==================== Globals for matchers, histories ====================
tick = 0
er_history = collections.deque(maxlen=settings.er_hist_size)
matcher_cache = {'timestamp': None, 'matcher': None}
matcher_executor = ThreadPoolExecutor(max_workers=1)

# ==================== 10. LIVE TRADING (динамічний символ + per-symbol wallet) ====================
matcher_executor   = matcher_executor  # already defined
# X_history, y_history declared above

async def selector_task_loop():
    """Background selector task: periodically scans markets and (best-effort)
    switches active_symbol to a better candidate when one is found _or_ when the current symbol
    produced no opens for a long time (selector_max_idle_minutes). This implementation is careful
    to honor selector_concurrency and selector_switch_margin settings and obtains switch_lock
    before mutating active_state to avoid races with run_live.
    """
    try:
        interval = getattr(settings, 'selector_interval', 60)
        while True:
            try:
                if not getattr(settings, 'selector_enabled', False):
                    await asyncio.sleep(interval)
                    continue

                # load markets once per cycle (async CCXT)
                try:
                    markets = await async_ex.load_markets()
                    symbols = []
                    for s, m in markets.items():
                        if not isinstance(s, str):
                            continue
                        if getattr(settings, 'scan_usdt_only', True) and not s.endswith('/USDT'):
                            continue
                        if getattr(settings, 'skip_leveraged_tokens', True):
                            name = s.upper()
                            if any(x in name for x in ('BULL', 'BEAR', 'UP', 'DOWN', '3L', '3S', '1D')):
                                continue
                        symbols.append(s)
                except Exception:
                    symbols = list(await async_ex.fetch_markets()) if hasattr(async_ex, 'fetch_markets') else []
                # filter out excluded bases (e.g., WAVES)
                try:
                    symbols = [s for s in symbols if isinstance(s, str) and s.split('/')[0].upper() not in EXCLUDED_BASES]
                except Exception:
                    pass

                if not symbols:
                    await asyncio.sleep(interval)
                    continue

                sem = asyncio.Semaphore(getattr(settings, 'selector_concurrency', 6))
                tasks = [asyncio.create_task(quick_score_for_symbol(sym, sem)) for sym in symbols]
                results = []
                # gather as they complete but with a timeout per whole batch
                done, pending = await asyncio.wait(tasks, timeout=30)
                for d in done:
                    try:
                        res = d.result()
                        if res:
                            results.append(res)
                    except Exception:
                        pass
                for p in pending:
                    try:
                        p.cancel()
                    except Exception:
                        pass

                if not results:
                    await asyncio.sleep(interval)
                    continue

                # choose top candidate by quick score
                results.sort(key=lambda x: x[1], reverse=True)
                best_sym, best_score, best_meta = results[0]
                # compute current symbol quick score (best-effort)
                cur_sym = active_state.get('active_symbol', settings.symbol)
                current_score = 0.0
                try:
                    current_score = await get_current_quick_score(cur_sym)
                except Exception:
                    current_score = 0.0

                # Decision logic:
                # - If best_score > current_score * (1 + selector_switch_margin) -> switch
                # - OR if current symbol produced no opens (wallet empty) or idle for long -> allow switch to any better candidate
                switch_margin = getattr(settings, 'selector_switch_margin', 0.05)
                allow_switch = False

                with state_lock:
                    wallet = active_state.get('wallets', {}).get(cur_sym, {'position':0.0,'trades':[]})

                # determine idle: no trades OR last trade older than max idle minutes
                try:
                    now_ts = time.time()
                    idle_minutes = float('inf')
                    # Prefer time since symbol entered NO-OPEN state (position became 0)
                    try:
                        if wallet and float(wallet.get('position', 0.0)) == 0.0:
                            pos_change = wallet.get('position_changed_ts', None)
                            if pos_change:
                                idle_minutes = (now_ts - float(pos_change)) / 60.0
                            else:
                                # fallback to last trade timestamp (older behavior)
                                if wallet and wallet.get('trades'):
                                    last = wallet['trades'][-1]
                                    last_trade_ts = float(last.get('time_ts') or 0.0)
                                    if last_trade_ts:
                                        idle_minutes = (now_ts - last_trade_ts) / 60.0
                        else:
                            # there is an open position -> not idle
                            idle_minutes = 0.0
                    except Exception:
                        idle_minutes = float('inf')
                except Exception:
                    idle_minutes = float('inf')

                if best_score > current_score * (1.0 + switch_margin):
                    allow_switch = True
                else:
                    try:
                        max_idle = getattr(settings, 'selector_max_idle_minutes', 5)
                        if idle_minutes is not None and idle_minutes >= float(max_idle):
                            allow_switch = True
                            log.info(f"Selector: allowing unconditional switch because {cur_sym} idle for {idle_minutes:.2f} min (threshold {max_idle})")
                        elif best_score > current_score * (1.0 + max(0.01, switch_margin*0.6)):
                            allow_switch = True
                    except Exception:
                        pass

                
                # --- Prevent switching while there's an open position on the current symbol ---
                try:
                    with state_lock:
                        cur_wallet = active_state.get('wallets', {}).get(cur_sym, {'position':0.0,'trades':[]})
                        if cur_wallet.get('position', 0.0) != 0.0:
                            # there's an open position: do not switch symbols until it's closed
                            allow_switch = False
                            log.info(f"Selector: will NOT switch away from {cur_sym} because open position exists (pos={cur_wallet.get('position')})")
                except Exception:
                    pass

                if allow_switch:
                    try:
                        # if best candidate is current symbol, rotate to next configured symbol (or markets list)
                        if best_sym == cur_sym:
                            sym_list = list(getattr(settings, 'symbols', [])) or symbols
                            # filter excluded bases
                            try:
                                sym_list = [s for s in sym_list if s and s.split('/')[0].upper() not in EXCLUDED_BASES]
                            except Exception:
                                pass
                            if sym_list:
                                try:
                                    idx = sym_list.index(cur_sym)
                                    cand = sym_list[(idx + 1) % len(sym_list)]
                                except Exception:
                                    cand = sym_list[0]
                                best_sym = cand
                        if best_sym and best_sym != cur_sym:
                            try:
                                with state_lock:
                                    # verify market availability before switching
                                    tdf = safe_fetch_df_sync(best_sym, settings.timeframes[0], 1)
                                    if tdf is None or len(tdf) == 0:
                                        try:
                                            log.debug('Selector: skipping switch to %s because OHLCV missing', best_sym)
                                        except Exception:
                                            pass
                                    else:
                                        active_state['active_symbol'] = best_sym
                                    if 'wallets' not in active_state:
                                        active_state['wallets'] = {}
                                    if best_sym not in active_state['wallets']:
                                        active_state['wallets'][best_sym] = {'position': 0.0, 'trades': []}
                                log.info(f"Selector switched active_symbol -> {best_sym} (cur_score={current_score:.4f} best={best_score:.4f})")
                            except Exception as e:
                                log.debug(f"Failed to switch symbol: {e}")
                    except Exception as e:
                        log.debug(f"Selector switch block error: {e}")
                # heartbeat sleep
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("Selector task error: %s", e)
                await asyncio.sleep(5)
    except Exception:
        return

async def run_live():
    """
    Simplified, robust run_live loop that preserves main behavior but avoids complex nested indentation bugs.
    This version is defensive: it logs errors, keeps wallets in active_state, and uses existing helpers for opens/closes.
    """
    global last_retrain_ts, retrain_interval, X_history, y_history
    global matcher_cache, matcher_executor, er_history

    # ensure async lock for selector / switching exists
    if active_state.get('switch_lock') is None:
        active_state['switch_lock'] = asyncio.Lock()

    # start selector task if enabled
    if active_state.get('selector_task') is None and settings.selector_enabled:
        try:
            active_state['selector_task'] = asyncio.create_task(selector_task_loop())
            log.info("Selector task started")
        except Exception as e:
            log.warning(f"Could not start selector task: {e}")

    if not hasattr(run_live, 'pos'):
        run_live.pos   = 0.0
        run_live.trade = None

    def ensure_wallet(sym):
        with state_lock:
            if sym not in active_state['wallets']:
                active_state['wallets'][sym] = {'position': 0.0, 'trades': []}

    # continuous loop
    while True:
        t0 = asyncio.get_event_loop().time()
        try:
            sym = active_state.get('active_symbol', settings.symbol)
            ensure_wallet(sym)
            wallet = active_state['wallets'][sym]

            # fetch OHLCV and basic metrics (robust fetch)
            try:
                df = await OHLCV.async_fetch(sym, settings.timeframes[0], settings.history_limit)
            except Exception as e:
                log.warning(f"run_live: OHLCV fetch failed for {sym}: {e}")
                await asyncio.sleep(1)
                continue

            if df is None or len(df) < 2:
                await asyncio.sleep(1)
                continue

            price = float(df['c'].iloc[-1])
            prev_price = float(df['c'].iloc[-2])
            vol = compute_volatility(df['c'].values, settings.vol_window)
            m = settings.wnd_min + (0 if vol < settings.target_vol else settings.wnd_step)
            mom = float(df['c'].iloc[-1] / df['c'].iloc[-(settings.adx_window+1)] - 1) if len(df) > settings.adx_window else 0.0

            # liquidity and marketcap (best-effort)
            liquidity_usd, liquidity_score = 0.0, 0.0
            market_cap = None
            try:
                ob = await async_ex.fetch_order_book(sym, 200)
                liquidity_usd, liquidity_score = compute_liquidity_score_from_orderbook(ob, price)
            except Exception:
                pass
            try:
                market_cap = fetch_marketcap_coingecko(sym)
            except Exception:
                market_cap = None

            # pattern matcher (constructed safely)
            pm = None
            try:
                loop = asyncio.get_running_loop()
                pm = await loop.run_in_executor(matcher_executor, PatternMatcherAE, df, m)
                matcher_cache.update({'matcher': pm, 'timestamp': df['ts'].iloc[-1] if 'ts' in df.columns else pd.Timestamp.now(tz='Europe/Kyiv')})
            except Exception:
                try:
                    pm = PatternMatcherAE(df, m)
                except Exception:
                    pm = None

            er_raw, ae_conf = 0.0, 0.0
            try:
                if pm:
                    er_raw, ae_conf = pm.match()
            except Exception:
                er_raw, ae_conf = 0.0, 0.0

            # blended ER
            alpha, beta = 0.75, 0.25
            er = alpha * er_raw * max(ae_conf, settings.min_ae_confidence) + beta * mom
            er_history.append(er)

            # breakout detection
            N = 20
            chan_high = df['h'].rolling(N).max().iloc[-2] if len(df) >= N+1 else df['h'].max()
            chan_low  = df['l'].rolling(N).min().iloc[-2] if len(df) >= N+1 else df['l'].min()
            atr = compute_atr(df, window=14)
            buf = settings.atr_multiplier_open * atr
            breakout_long  = price > chan_high + buf
            breakout_short = price < chan_low  - buf

            # adaptive parameters and thresholds
            adaptive = compute_adaptive_params(vol, liquidity_score=liquidity_score, market_cap=market_cap)
            thr = adaptive['score_threshold']
            local_min_time_between = adaptive['min_time_between_trades']
            local_stop_loss_atr_mult = adaptive['stop_loss_atr_mult']

            # scoring
            f_er_long  = safe_norm(er, -0.1, 0.1)
            f_er_short = safe_norm(-er, -0.1, 0.1)
            f_breakout_long  = float(breakout_long)
            f_breakout_short = float(breakout_short)
            f_vol = float(df['v'].iloc[-1] > df['v'].iloc[-settings.vol_window:].mean() * 1.1) if len(df) >= settings.vol_window else 0.0
            mom_norm = np.tanh(mom * 50) * 0.5 + 0.5

            score_long  = (settings.w_breakout * f_breakout_long +
                           settings.w_er * f_er_long +
                           settings.w_mom * (mom_norm if mom>0 else 0) +
                           settings.w_vol * f_vol)

            score_short = (settings.w_breakout * f_breakout_short +
                           settings.w_er * f_er_short +
                           settings.w_mom * (1-mom_norm if mom<0 else 0) +
                           settings.w_vol * f_vol)

            if ae_conf < adaptive['min_ae_confidence']:
                penal = 0.15 * (adaptive['min_ae_confidence'] - ae_conf)
                score_long  = max(0.0, score_long - penal)
                score_short = max(0.0, score_short - penal)

            # sizing (simple)
            f = 0.02
            amt = max(0.000001, round(settings.base_usd * f / max(price, 1e-8), 6))

            # last open guard
            last_open_ok = True
            try:
                if wallet.get('trades'):
                    last_open_ts = wallet['trades'][-1].get('time_ts', None)
                    if last_open_ts and (time.time() - float(last_open_ts)) < local_min_time_between:
                        last_open_ok = False
            except Exception:
                last_open_ok = True

            # OPEN logic (dry-run support)
            if wallet.get('position', 0.0) == 0.0 and last_open_ok:
                # compute vol spike & very short-term EMA to detect start of move
                vol_spike = False
                try:
                    recent_v_mean = df['v'].iloc[-settings.vol_window:].mean() if len(df) >= settings.vol_window else df['v'].mean()
                    vol_spike = df['v'].iloc[-1] > recent_v_mean * 1.35
                except Exception:
                    vol_spike = False

                # small-window EMA for early detection
                try:
                    ema_short = df['c'].ewm(span=3, adjust=False).mean().iloc[-1]
                except Exception:
                    ema_short = price

                # immediate price change over last candle (fast momentum)
                price_change = (price / prev_price - 1.0) if prev_price and prev_price > 0 else 0.0

                # early-entry heuristic: noticeable one-bar momentum + volume spike + price above ema_short
                early_long = (price_change > max(0.002, 0.25 * (atr / max(price, 1e-8)))) and vol_spike and (price > ema_short)
                early_short = (price_change < -max(0.002, 0.25 * (atr / max(price, 1e-8)))) and vol_spike and (price < ema_short)

                # apply adaptive threshold adjustments (easier opens when vol low)
                effective_thr = thr * (1.0 - 0.25 * adaptive.get('vol_factor', 0.0))

                # final open decision: accept breakouts OR early-entry if score reasonably high
                if score_long >= thr or (early_long and score_long >= (effective_thr * 0.7)):
                    notional_open = float(round(amt * price))
                    tr = execute_open_trade(sym, wallet, 'Long', amt, price,
                                            extra={'open_idx': len(df)-1, 'atr': atr, 'er_raw': er_raw, 'ae_conf': ae_conf, 'notional_usd': notional_open},
                                            dry=settings.dry_run)
                    run_live.pos = amt
                    log.info(f"OPEN Long {sym} size={amt} price={price} early={early_long} score={score_long:.3f} thr={thr:.3f}")
                elif score_short >= thr or (early_short and score_short >= (effective_thr * 0.7)):
                    notional_open = float(round(amt * price))
                    tr = execute_open_trade(sym, wallet, 'Short', amt, price,
                                            extra={'open_idx': len(df)-1, 'atr': atr, 'er_raw': er_raw, 'ae_conf': ae_conf, 'notional_usd': notional_open},
                                            dry=settings.dry_run)
                    run_live.pos = -amt
                    log.info(f"OPEN Short {sym} size={amt} price={price} early={early_short} score={score_short:.3f} thr={thr:.3f}")

            # CLOSE logic
            if wallet.get('trades'):
                tr = wallet['trades'][-1]
                price_now = float(df['c'].iloc[-1])
                atr = tr.get('atr', compute_atr(df, window=14))
                trailing_triggered = False
                stop_triggered = False
                try:
                    if wallet.get('position', 0.0) > 0:
                        init_stop = tr['price'] - local_stop_loss_atr_mult * atr
                        # maintain explicit high-water mark for trailing (only increases)
                        tr['trail_high'] = max(tr.get('trail_high', tr['price']), price_now)
                        # arm trailing only after a minimal profit threshold to avoid early noise
                        try:
                            armed = tr.get('trail_armed', False) or (price_now >= tr['price'] * (1.0 + float(getattr(settings, 'min_profit_to_trail_pct', 0.001))))
                            tr['trail_armed'] = armed
                        except Exception:
                            tr['trail_armed'] = tr.get('trail_armed', False)
                        # compute two triggers: ATR-based and percentage-based
                        trigger_atr = tr['trail_high'] - settings.trailing_atr_mult * atr
                        trigger_pct = tr['trail_high'] * (1.0 - settings.trailing_pct)
                        # trailing triggers if either threshold is breached (more robust)
                        trailing_triggered = False
                        try:
                            if tr.get('trail_armed', False):
                                trailing_triggered = (price_now <= trigger_atr) or (price_now <= trigger_pct)
                            else:
                                trailing_triggered = False
                        except Exception:
                            trailing_triggered = False
                        # debug logging for trailing diagnostics
                        try:
                            log.debug(f"TRAIL CHECK long: price_now={price_now:.6f}, trail_high={tr.get('trail_high')}, trigger_atr={trigger_atr:.6f}, trigger_pct={trigger_pct:.6f}, armed={tr.get('trail_armed')}, atr={atr:.6f}")
                        except Exception:
                            pass
                        stop_triggered = price_now <= init_stop
                    else:
                        init_stop = tr['price'] + local_stop_loss_atr_mult * atr
                        # maintain explicit low-water mark for trailing (only decreases)
                        tr['trail_low'] = min(tr.get('trail_low', tr['price']), price_now)
                        # arm trailing only after minimal profit threshold for shorts
                        try:
                            armed = tr.get('trail_armed', False) or (price_now <= tr['price'] * (1.0 - float(getattr(settings, 'min_profit_to_trail_pct', 0.001))))
                            tr['trail_armed'] = armed
                        except Exception:
                            tr['trail_armed'] = tr.get('trail_armed', False)
                        trigger_atr = tr['trail_low'] + settings.trailing_atr_mult * atr
                        trigger_pct = tr['trail_low'] * (1.0 + settings.trailing_pct)
                        trailing_triggered = False
                        try:
                            if tr.get('trail_armed', False):
                                trailing_triggered = (price_now >= trigger_atr) or (price_now >= trigger_pct)
                            else:
                                trailing_triggered = False
                        except Exception:
                            trailing_triggered = False
                        # debug logging for trailing diagnostics (short)
                        try:
                            log.debug(f"TRAIL CHECK short: price_now={price_now:.6f}, trail_low={tr.get('trail_low')}, trigger_atr={trigger_atr:.6f}, trigger_pct={trigger_pct:.6f}, armed={tr.get('trail_armed')}, atr={atr:.6f}")
                        except Exception:
                            pass
                        stop_triggered = price_now >= init_stop
                except Exception:
                    trailing_triggered = False
                    stop_triggered = False

                time_exit = False
                try:
                    tr_time = tr.get('time')
                    if tr_time is None and 'time_ts' in tr:
                        tr_time = pd.to_datetime(tr['time_ts'], unit='s', utc=True).tz_convert('Europe/Kyiv')
                    age_minutes = (pd.Timestamp.now(tz='Europe/Kyiv') - tr_time).total_seconds() / 60.0 if tr_time is not None else 0.0
                    time_exit = age_minutes > settings.max_trade_minutes
                except Exception:
                    time_exit = False

                model_exit = False
                try:
                    feat = make_exit_feature(df.reset_index(), tr, er, ae_conf)
                    if last_retrain_ts and len(X_history) >= 1:
                        p_stop = float(exit_model.predict_proba(feat)[0,1])
                        if p_stop > settings.exit_prob_thr:
                            model_exit = True
                except Exception:
                    model_exit = False

                should_exit = trailing_triggered or stop_triggered or model_exit or time_exit
                if should_exit:
                    try:
                        pnl_scaled = execute_close_trade(sym, wallet, tr, price_now, df_for_feat=df, reason=('trailing' if trailing_triggered else ('stop' if stop_triggered else ('model' if model_exit else 'timeout'))), dry=settings.dry_run)
                        log.info(f"CLOSE {sym} pnl_scaled={pnl_scaled}")
                    except Exception as e:
                        log.warning(f"Close failed: {e}")

        except Exception as e:
            log.warning(f"run_live loop error: {e}", exc_info=True)

        # sleep to maintain ~60s loop cadence
        try:
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0, 60 - elapsed))
        except Exception:
            await asyncio.sleep(1)
BOT_TOKEN  = os.getenv('BOT_TOKEN',"7769105003:AAG5NwNm9cOodvF3gNSrx43hpaVNEqFufiI")
CHANNEL_ID = os.getenv('CHANNEL_ID',"-1002591018946")

if not BOT_TOKEN or not CHANNEL_ID:
    log.warning("BOT_TOKEN/CHANNEL_ID not provided — Telegram disabled")

def send_telegram(text: str):
    """
    Send message to Telegram in background with retries.
    Will attempt to send if BOT_TOKEN and CHANNEL_ID are present.
    """
    try:
        BOT_TOKEN = globals().get("BOT_TOKEN") or os.environ.get("BOT_TOKEN")
        CHANNEL_ID = globals().get("CHANNEL_ID") or os.environ.get("CHANNEL_ID")
    except Exception:
        BOT_TOKEN = None; CHANNEL_ID = None

    if not BOT_TOKEN or not CHANNEL_ID:
        logging.getLogger().debug("TG disabled or missing credentials, would send: %s", text)
        return

    def _send():
        import time as _t
        import requests
        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                    data={'chat_id': CHANNEL_ID, 'text': text, 'parse_mode': 'HTML'},
                    timeout=6
                )
                r.raise_for_status()
                logging.getLogger().info("Telegram sent (attempt %d)", attempt+1)
                return
            except Exception as e:
                logging.getLogger().warning("Telegram send attempt %d failed: %s", attempt+1, e)
                _t.sleep(1 + attempt*2)
        logging.getLogger().error("Telegram failed after retries")
    try:
        threading.Thread(target=_send, daemon=True).start()

    except Exception as e:
        logging.getLogger().exception("Telegram background thread failed: %s", e)

def format_open_signal(symbol, direction, price, size, er, vol, direction_bool, vol_spike):
    emoji = "🔼" if direction == 'LONG' else "🔽"
    now = pd.Timestamp.now(tz='Europe/Kyiv').strftime('%H:%M')
    price_str = f"{price:.6f}"
    notional = int(round(size * price)) if size is not None else 0
    strength = min(abs(er) / 0.05, 1.0) if er is not None else 0.0
    msg = f"⏰ <b>{now}</b> | {emoji} <b>OPEN {direction} {symbol}</b>\n"
    msg += f"Price: <code>{price_str}</code>\n"
    msg += f"Notional (est): <code>{notional}</code>\n"
    msg += f"ER (est): <b>{(er*100 if er is not None else 0):.2f}%</b> • Strength: <b>{strength:.2f}</b>\n"
    if vol_spike:
        msg += "<b>🚀 Volume Spike!</b>\n"
    return msg

SYMBOL, TF, LIMIT, REFRESH = (
    settings.symbol,
    settings.timeframes[1],
    settings.chart_limit,
    settings.chart_refresh_ms
)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           title=f"📈 {SYMBOL} ({TF})")
app.layout = html.Div(
    style={"position":"absolute","top":0,"left":0,"right":0,"bottom":0,
           "display":"flex","flexDirection":"column","overflow":"hidden","margin":0,"padding":0},
    children=[
        dcc.Graph(id="live-chart",
              config={"locale":"uk-UA","displayModeBar":False,"displaylogo":False},
              style={"flexGrow":1,"minHeight":0,"margin":0,"padding":0}
        ),
        html.Div(style={"flexShrink":0,"maxHeight":"26%","overflow":"auto","backgroundColor":"#071225","padding":"8px"},
             children=html.Div(id="metrics-cards-container", style={"display":"flex","gap":"8px","alignItems":"stretch","padding":"8px"},
                 children=[
                     dbc.Card([
                         dbc.CardBody([
                             html.Div("Equity", style={"fontSize":"12px","color":"#a9cbdc"}),
                             html.Div(id="card-equity", style={"fontSize":"20px","fontWeight":"700"})
                         ])
                     ], style={"flex":"1","backgroundColor":"#071b2a","border":"1px solid rgba(255,255,255,0.03)"}),

                     dbc.Card([
                         dbc.CardBody([
                             html.Div("Total Realized PnL", style={"fontSize":"12px","color":"#a9cbdc"}),
                             html.Div(id="card-realized", style={"fontSize":"18px","fontWeight":"700"})
                         ])
                     ], style={"width":"220px","backgroundColor":"#052e20"}),

                     dbc.Card([
                         dbc.CardBody([
                             html.Div("Open Position", style={"fontSize":"12px","color":"#a9cbdc"}),
                             html.Div(id="card-open-pos", style={"fontSize":"16px","fontWeight":"700"})
                         ])
                     ], style={"width":"220px","backgroundColor":"#10202a"}),

                     dbc.Card([
                         dbc.CardBody([
                             html.Div("Unrealized PnL (current)", style={"fontSize":"12px","color":"#a9cbdc"}),
                             html.Div(id="card-unrealized", style={"fontSize":"16px","fontWeight":"700"})
                         ])
                     ], style={"width":"220px","backgroundColor":"#071225"}),

                     dbc.Card([
                         dbc.CardBody([
                             html.Div("Win Rate", style={"fontSize":"12px","color":"#a9cbdc"}),
                             html.Div(id="card-winrate", style={"fontSize":"16px","fontWeight":"700"})
                         ])
                     ], style={"width":"140px","backgroundColor":"#071225"})
                 ]
             )
        ),

        dcc.Interval(id="interval", interval=REFRESH, n_intervals=0),
        dcc.Store(id="wallet-store", data={"active_symbol": settings.symbol, "wallets": {settings.symbol: {"position":0.0,"trades":[]}}})
    ]
)

# ------------------ CALLBACK 1: основний (figure, metrics, wallet) ------------------
@app.callback(
    Output("live-chart", "figure"),
    Output("metrics-cards-container", "children"),
    Output("wallet-store", "data"),
    Input("interval", "n_intervals"),
    State("wallet-store", "data"),
    prevent_initial_call=True
)
def update(n, store):
    """
    Full updated callback:
     - keeps all existing indicators / traces
     - adds colored block marker on the last candle with position type and PnL
     - fixes close logic to ensure exit_price/exit_time/pnl/pnl_scaled are set (and saved)
     - improves metrics table content and visualized fields
    """
    global tick, exit_model, last_retrain_ts, retrain_interval, X_history, y_history
    global er_history, matcher_cache, matcher_executor

    # --- BEGIN: non-destructive store/wallet merge (inserted by assistant) ---
    # This block merges server-side wallets into client store without overwriting existing history.
    if store is None:
        store = {}
    store.setdefault('wallets', {})
    # derive active symbol from server state
    try:
        with state_lock:
            active_symbol = active_state.get('active_symbol', settings.symbol)
            active_state.setdefault('wallets', {})
            active_state['wallets'].setdefault(active_symbol, {'position': 0.0, 'trades': []})
    except Exception:
        # fallback if state or lock not available
        active_symbol = active_state.get('active_symbol', settings.symbol) if isinstance(active_state, dict) else settings.symbol

    # Merge server wallets into client store non-destructively
    try:
        with state_lock:
            server_wallets = active_state.get('wallets', {})
            for s, w in server_wallets.items():
                if s not in store['wallets']:
                    store['wallets'][s] = {'position': float(w.get('position', 0.0)), 'trades': list(w.get('trades', []))}
                else:
                    client_w = store['wallets'][s]
                    client_trades = client_w.get('trades', []) or []
                    client_times = set([str(t.get('time')) for t in client_trades if t.get('time') is not None])
                    for tr in (w.get('trades') or []):
                        tr_time = tr.get('time')
                        tr_key = str(tr_time)
                        if tr_key not in client_times:
                            client_trades.append(tr)
                            client_times.add(tr_key)
                    client_w['trades'] = client_trades
                    if not client_w.get('position'):
                        client_w['position'] = float(w.get('position', 0.0))
                    store['wallets'][s] = client_w
    except Exception:
        # if state_lock or active_state missing, skip merge
        pass

    # If active symbol missing trades in store, try to load DB trades and merge
    try:
        if active_symbol not in store['wallets'] or not store['wallets'][active_symbol].get('trades'):
            loaded = load_trades_from_db(active_symbol)
            if loaded and loaded.get('trades'):
                existing = store['wallets'].get(active_symbol, {'position':0.0,'trades':[]})
                existing_trades = existing.get('trades', [])
                existing_times = set([str(t.get('time')) for t in existing_trades if t.get('time') is not None])
                for t in loaded.get('trades', []):
                    key = str(t.get('time'))
                    if key not in existing_times:
                        existing_trades.append(t)
                        existing_times.add(key)
                existing['trades'] = existing_trades
                existing['position'] = float(existing.get('position', 0.0)) or float(loaded.get('position', 0.0))
                store['wallets'][active_symbol] = existing
    except Exception:
        pass

    store['active_symbol'] = active_symbol
    try:
        for w in store['wallets'].values():
            ensure_trade_timestamps(w.get('trades', []))
    except Exception:
        pass

    wallet = store['wallets'].setdefault(active_symbol, {'position':0.0,'trades':[]})
    # --- END: non-destructive store/wallet merge ---

    if store is None:
        store = {"active_symbol": settings.symbol, "wallets": {settings.symbol: {"position":0.0,"trades":[]}}}

    with state_lock:
        active_symbol = active_state.get('active_symbol', settings.symbol)
        if 'wallets' not in store:
            store['wallets'] = {}
        if active_symbol not in store['wallets']:
            # create placeholder then try to load persisted trades from DB
            store['wallets'][active_symbol] = {"position":0.0, "trades":[]}
            try:
                loaded = load_trades_from_db(active_symbol)
                if loaded and loaded.get('trades'):
                    store['wallets'][active_symbol] = loaded
            except Exception:
                pass
        store['active_symbol'] = active_symbol

    # parse timestamps from store (Dash stores ISO strings)
    try:
        ensure_trade_timestamps(store['wallets'][active_symbol].get('trades', []))
    except Exception:
        pass

    # tick counter
    try:
        tick += 1
    except NameError:
        tick = 1

    # Periodic AE retrain (non-blocking best-effort)
    if tick % settings.dash_train_interval == 0:
        try:
            dfh = OHLCV.sync_fetch(active_symbol, settings.timeframes[0], settings.history_limit)
            # cooldown guard to avoid frequent retrains (seconds)
            try:
                now_ts = time.time()
                if last_retrain_ts is not None and (now_ts - last_retrain_ts) < getattr(settings, 'retrain_cooldown', 3600):
                    # skip this retrain due to cooldown
                    log.info("Skipping AE retrain due to cooldown")
                    dfh = None
                else:
                    # will proceed; update last_retrain_ts after successful retrain
                    pass
            except Exception:
                pass

            
            if dfh is not None and len(dfh) > 0:
                schedule_retrain(dfh, settings.wnd_min, f"conv_ae_m{settings.wnd_min}.pth")
            else:
                try:
                    log.warning('Dash autoencoder retrain skipped: no OHLCV for %s', active_symbol)
                except Exception:
                    pass

        except Exception as e:
            log.warning(f"Dash autoencoder retrain failed: {e}")

    # INITIAL small-run protection
    if n is None or n <= 1:
        try:
            _tmp_df0 = safe_fetch_df_sync(active_symbol, TF, LIMIT)
            if _tmp_df0 is None or len(_tmp_df0) == 0:
                try:
                    log.warning('safe_fetch: OHLCV returned None/empty for active_symbol, TF, LIMIT')
                except Exception:
                    pass
                import pandas as _pd
                _tmp_df0 = _pd.DataFrame(columns=['ts','o','h','l','c','v'])
            df0 = _tmp_df0.set_index('ts')
        except Exception as e:
            log.warning(f"Dash initial OHLCV failed for {active_symbol}: {e}")
            cards = build_metrics_cards([])
            return go.Figure(), cards, store

        fig0 = go.Figure(go.Ohlc(
            x=df0.index,
            open=df0['o'], high=df0['h'], low=df0['l'], close=df0['c'],
            increasing_line_color='green', decreasing_line_color='red'
        ))
        fig0.update_layout(template='plotly_dark', autosize=True,
                           margin=dict(l=60, r=140, t=120, b=40),
                           xaxis=dict(rangeslider_visible=False, automargin=True),
                           yaxis=dict(autorange=True, automargin=True),
                           hovermode='x unified')
        init_metrics = {
            "equity": int(round(settings.base_usd)),
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "last_trade_pnl": 0.0,
            "open_position": "Neutral",
            "open_size": 0.0,
            "trades_count": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "liq_usd": 0,
            "liq_score": 0.0,
            "signal_strength": 0.0,
            "er": 0.0,
            "ae_conf": 0.0
        }
        try:
            sanitize_annotations(fig)
        except Exception:
            pass

        cards = build_metrics_cards([init_metrics])
        return fig0, cards, store

    # --- fetch OHLC data for chart ---
    try:
        _tmp_df = safe_fetch_df_sync(active_symbol, TF, LIMIT)
        if _tmp_df is None or len(_tmp_df) == 0:
            try:
                log.warning('safe_fetch: OHLCV returned None/empty for active_symbol, TF, LIMIT')
            except Exception:
                pass
            import pandas as _pd
            _tmp_df = _pd.DataFrame(columns=['ts','o','h','l','c','v'])
        df = _tmp_df.set_index('ts')
    except Exception as e:
        log.warning(f"Dash fetch OHLCV failed for {active_symbol}: {e}")
        cards = build_metrics_cards([])
        return go.Figure(), cards, store

    ft, lt = df.index[0], df.index[-1]
    price = float(df['c'].iloc[-1])

    # build / refresh pattern matcher if needed
    pm = None
    need_rebuild = matcher_cache.get('timestamp') is None or pd.to_datetime(lt) > pd.to_datetime(matcher_cache.get('timestamp'))
    if need_rebuild:
        try:
            fut = matcher_executor.submit(PatternMatcherAE, df.reset_index(), settings.wnd_min)
            pm = fut.result(timeout=10)
            matcher_cache.update({'matcher': pm, 'timestamp': lt})
        except Exception as e:
            log.debug(f"Callback PatternMatcherAE build failed or timed out: {e}")
            pm = matcher_cache.get('matcher')
    else:
        pm = matcher_cache.get('matcher')

    # compute ER and AE confidence
    try:
        if pm is not None:
            match_res = pm.match()
            if isinstance(match_res, tuple) and len(match_res) == 2:
                er_raw, ae_conf = match_res
            else:
                er_raw, ae_conf = float(match_res), 0.0
        else:
            er_raw, ae_conf = 0.0, 0.0
    except Exception:
        er_raw, ae_conf = 0.0, 0.0

    mom = float(df['c'].iloc[-1] / df['c'].iloc[-(settings.adx_window+1)] - 1) if len(df) > settings.adx_window else 0.0
    alpha, beta = 0.75, 0.25
    er = alpha * er_raw * max(ae_conf, settings.min_ae_confidence) + beta * mom
    er_history.append(er)

    vol = compute_volatility(df['c'].values, settings.vol_window)
    m = settings.wnd_min + (0 if vol < settings.target_vol else settings.wnd_step)

    # liquidity & marketcap
    liquidity_usd, liquidity_score = 0.0, 0.0
    try:
        ob = sync_ex.fetch_order_book(active_symbol, 200)
        liquidity_usd, liquidity_score = compute_liquidity_score_from_orderbook(ob, price)
    except Exception as e:
        log.debug(f"Order book fetch failed (dash): {e}")
    try:
        market_cap = fetch_marketcap_coingecko(active_symbol)
    except Exception:
        market_cap = None

    adaptive = compute_adaptive_params(vol, liquidity_score=liquidity_score, market_cap=market_cap)
    local_score_thr = adaptive['score_threshold']
    local_min_time_between = adaptive['min_time_between_trades']
    local_atr_open = adaptive['atr_multiplier_open']
    local_open_soft = adaptive['open_softness_factor']
    local_min_ae = adaptive['min_ae_confidence']
    local_stop_loss_atr_mult = adaptive['stop_loss_atr_mult']
    thr = local_score_thr

    # breakout / signals
    N = 20
    chan_high = df['h'].rolling(N).max().iloc[-2] if len(df) >= N+1 else df['h'].max()
    chan_low  = df['l'].rolling(N).min().iloc[-2] if len(df) >= N+1 else df['l'].min()
    atr = compute_atr(df, window=14)
    buf = local_atr_open * atr
    breakout_long  = price > chan_high + buf
    breakout_short = price < chan_low - buf

    # score components
    if len(er_history) >= 20:
        er_hi = np.percentile(er_history, 90)
        er_lo = np.percentile(er_history, 10)
    else:
        er_hi, er_lo = max(er, 1e-6), min(er, -1e-6)
    f_er_long  = safe_norm(er, er_lo, er_hi)
    f_er_short = safe_norm(-er, -er_hi, -er_lo)
    mom_norm = np.tanh(mom * 50) * 0.5 + 0.5
    vol_spike = df['v'].iloc[-1] > df['v'].iloc[-settings.vol_window:].mean() * 1.1 if len(df) >= settings.vol_window else False

    score_long  = (settings.w_breakout * float(breakout_long) +
                   settings.w_er * f_er_long +
                   settings.w_mom * (mom_norm if mom>0 else 0) +
                   settings.w_vol * float(vol_spike))

    score_short = (settings.w_breakout * float(breakout_short) +
                   settings.w_er * f_er_short +
                   settings.w_mom * (1-mom_norm if mom<0 else 0) +
                   settings.w_vol * float(vol_spike))

    if ae_conf < local_min_ae:
        penal = 0.15 * (local_min_ae - ae_conf)
        score_long  = max(0.0, score_long - penal)
        score_short = max(0.0, score_short - penal)

    # sizing (simplified)
    f = 1.0
    try:
        if len(X_history) >= 1 and last_retrain_ts is not None:
            probs = exit_model.predict_proba(np.vstack(X_history))[:,1]
            win_rate = (probs > 0.5).mean() * 100
            pnl_arr = np.array([t for t in er_history if isinstance(t,(float,int))]) if er_history else np.array([])
            avg_win = pnl_arr[pnl_arr>0].mean() if any(pnl_arr>0) else 0.0
            avg_loss = abs(pnl_arr[pnl_arr<0].mean()) if any(pnl_arr<0) else 0.001
            if avg_loss == 0:
                avg_loss = 0.001
            kf = (win_rate/100) - ((1-win_rate/100) / ((avg_win/(avg_loss+1e-12)) + 1e-12))
            f = max(min(kf,0.1),0.01)
    except Exception:
        f = 1.0
    f = min(f, adaptive['kelly_cap'])
    amt = settings.base_usd * f / max(price, 1e-8)
    amt = round(amt, 6)

    # sync wallet from active_state (if live trainer updated it)
    with state_lock:
        if active_symbol in active_state.get('wallets', {}):
            store['wallets'][active_symbol] = active_state['wallets'][active_symbol]
        wallet = store['wallets'][active_symbol]

    # last_open_ok check
    last_open_ok = True
    if wallet.get('trades'):
        last_ts = wallet['trades'][-1].get('time_ts')
        if last_ts and (time.time() - float(last_ts)) < local_min_time_between:
            last_open_ok = False

    # simulate opens for UI only (do not send orders here)
    with state_lock:
        if wallet.get('position', 0.0) == 0.0 and last_open_ok:
            if score_long >= thr:
                tr = {
                    'time': lt,
                    'time_ts': time.time(),
                    'price': price,
                    'type': 'Long',
                    'pnl': None,
                    'pnl_scaled': None,
                    'open_idx': len(df)-1,
                    'er': er,
                    'er_raw': er_raw,
                    'ae_conf': ae_conf,
                    'size': float(amt),
                    'notional_usd': float(int(round(amt * price))),
                    'trail_base': price,
                    'atr': atr
                }
                wallet['position'] = amt
                wallet['trades'].append(tr)
                try:
                    send_telegram(format_open_signal(active_symbol,'LONG',price,amt,er,vol_spike,True,vol_spike))
                except Exception:
                    pass
            elif score_short >= thr:
                tr = {
                    'time': lt,
                    'time_ts': time.time(),
                    'price': price,
                    'type': 'Short',
                    'pnl': None,
                    'pnl_scaled': None,
                    'open_idx': len(df)-1,
                    'er': er,
                    'er_raw': er_raw,
                    'ae_conf': ae_conf,
                    'size': float(amt),
                    'notional_usd': float(int(round(amt * price))),
                    'trail_base': price,
                    'atr': atr
                }
                wallet['position'] = -amt
                wallet['trades'].append(tr)
                try:
                    send_telegram(format_open_signal(active_symbol,'SHORT',price,amt,er,vol_spike,False,vol_spike))
                except Exception:
                    pass

    # CLOSE logic (ensure exit values are set and saved)
    with state_lock:
        if wallet.get('trades'):
            tr = wallet['trades'][-1]
            price_now = float(df['c'].iloc[-1])
            atr = tr.get('atr', compute_atr(df, window=14))
            if wallet.get('position',0.0) > 0:
                tr['trail_base'] = max(tr.get('trail_base', tr['price']), price_now)
                trigger = tr['trail_base'] - settings.trailing_atr_mult * atr
                trailing_triggered = price_now <= trigger
                stop_triggered = price_now <= (tr['price'] - local_stop_loss_atr_mult * atr)
            else:
                tr['trail_base'] = min(tr.get('trail_base', tr['price']), price_now)
                trigger = tr['trail_base'] + settings.trailing_atr_mult * atr
                trailing_triggered = price_now >= trigger
                stop_triggered = price_now >= (tr['price'] + local_stop_loss_atr_mult * atr)

            try:
                tr_time = tr.get('time')
                if tr_time is None:
                    if 'time_ts' in tr:
                        tr_time = pd.to_datetime(tr['time_ts'], unit='s', utc=True).tz_convert('Europe/Kyiv')
                    else:
                        tr_time = pd.Timestamp.now(tz='Europe/Kyiv')
                age_minutes = (pd.Timestamp.now(tz='Europe/Kyiv') - tr_time).total_seconds() / 60.0
            except Exception:
                age_minutes = 0.0
            time_exit = age_minutes > settings.max_trade_minutes

            # model exit check (if exit_model trained)
            model_exit = False
            try:
                feat = make_exit_feature(df.reset_index(), tr, er, ae_conf)
                if last_retrain_ts and len(X_history) >= 1:
                    p_stop = float(exit_model.predict_proba(feat)[0,1])
                    if p_stop > settings.exit_prob_thr:
                        model_exit = True
            except Exception:
                model_exit = False

            should_exit = trailing_triggered or stop_triggered or model_exit or time_exit
            if should_exit:
                try:
                    # execute_close_trade should update tr with exit_price/exit_time/pnl/pnl_scaled when not dry
                    pnl_scaled = execute_close_trade(
                        active_symbol,
                        wallet,
                        tr,
                        price_now,
                        df_for_feat=(df if 'df' in locals() else None),
                        reason=('trailing' if trailing_triggered else ('stop' if stop_triggered else ('model' if model_exit else 'timeout'))),
                        dry=settings.dry_run
                    )
                except Exception as e:
                    log.warning(f"Close failed: {e}")

                                # Ensure we still set exit fields on UI **only if they are not already set**
                try:
                    if tr.get('exit_price') is None and tr.get('pnl_scaled') is None:
                        tr['exit_price'] = float(price_now)
                        # prefer bar timestamp where price came from (last candle) so the exit marker stays pinned to that candle
                        try:
                            bar_ts = df.index[-1] if ('df' in locals() and hasattr(df, 'index') and len(df.index) > 0) else pd.Timestamp.now(tz='Europe/Kyiv')
                            tr['exit_time'] = pd.to_datetime(bar_ts)
                        except Exception:
                            tr['exit_time'] = pd.Timestamp.now(tz='Europe/Kyiv')

                        typ = tr.get('type', 'Long')
                        size_amt = float(tr.get('size', 0.0))
                        entry_price = float(tr.get('price') or tr.get('entry_price') or price_now)
                        pnl_usdt = (tr['exit_price'] - entry_price) * size_amt if typ == 'Long' else (entry_price - tr['exit_price']) * size_amt
                        not_usd = float(tr.get('notional_usd') or max(abs(size_amt) * entry_price, 1.0))
                        tr['pnl'] = float(pnl_usdt)
                        tr['pnl_scaled'] = float(pnl_usdt * (settings.base_usd / max(not_usd, 1e-8)))

                        # snapshot closed info and mark marker-removed so UI treats closed trade as static

                        # Persist closed trade in DB (best-effort)
                        try:
                            save_trade_to_db(tr)
                        except Exception:
                            log.debug("Save to DB failed", exc_info=True)
                except Exception as e:
                    log.warning("UI-close fallback failed: %s", e)

    # ------------------ Build figure (traces, annotations, existing indicators preserved) ------------------
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', autosize=True,
                      margin=dict(l=60, r=140, t=180, b=40),
                      xaxis=dict(rangeslider_visible=False, automargin=True),
                      hovermode='x unified', showlegend=False,
                      title={'text': f"<b>{active_symbol}</b> — {TF} | {price:.2f}<br><span style='font-size:12px'>ER: {er:.3f} | AE: {ae_conf:.3f} | Liq: ${int(adaptive.get('liquidity_usd_recent',0))} ({adaptive.get('liquidity_score',0.0):.3f}) | S_long:{score_long:.3f} S_short:{score_short:.3f}</span>", 'x':0.01, 'xanchor':'left'})

    # --- Long/Short percentage gauge (top center) ---
    try:
        ls_long = float(score_long) if 'score_long' in locals() else 0.0
        ls_short = float(score_short) if 'score_short' in locals() else 0.0
        if ls_long + ls_short > 0:
            ls_ratio = ls_long / (ls_long + ls_short)
        else:
            ls_ratio = 0.5

        # central gauge width from 0.32 to 0.68 paper coords (narrower pill)
        gx0, gx1 = 0.32, 0.68
        inner_start = gx0 + 0.01
        inner_end = gx1 - 0.01

        # reduce vertical footprint: smaller pill height so it doesn't crowd chart
        pill_y0 = 0.945
        pill_y1 = 0.975
        border_y0 = pill_y0
        border_y1 = pill_y1

        # compute split point
        long_x1 = inner_start + (inner_end - inner_start) * ls_ratio
        short_x0 = long_x1

        # subtle background pill
        fig.add_shape(type='rect', xref='paper', yref='paper', x0=gx0, x1=gx1, y0=pill_y0, y1=pill_y1,
                      fillcolor='rgba(255,255,255,0.02)', line={'width':0}, layer='above')

        # long fill (left) and short fill (right)
        fig.add_shape(type='rect', xref='paper', yref='paper', x0=inner_start, x1=long_x1, y0=pill_y0+0.002, y1=pill_y1-0.002,
                      fillcolor=f'rgba(0,200,120,{0.55 + 0.35*ls_ratio})', line={'width':0}, layer='above')
        fig.add_shape(type='rect', xref='paper', yref='paper', x0=short_x0, x1=inner_end, y0=pill_y0+0.002, y1=pill_y1-0.002,
                      fillcolor=f'rgba(220,60,60,{0.55 + 0.35*(1-ls_ratio)})', line={'width':0}, layer='above')

        # thin border on pill for contrast
        fig.add_shape(type='rect', xref='paper', yref='paper', x0=inner_start, x1=inner_end, y0=border_y0, y1=border_y1,
                      fillcolor='rgba(0,0,0,0)', line={'color':'rgba(255,255,255,0.12)','width':1.2}, layer='above')

        # compact donut (moved slightly down to avoid overlap with pill)
        try:
            fig.add_trace(go.Pie(labels=['Long','Short'],
                                 values=[ls_ratio, 1-ls_ratio],
                                 hole=0.6, sort=False, direction='clockwise',
                                 marker=dict(colors=['rgba(0,200,120,0.9)','rgba(220,60,60,0.9)']),
                                 textinfo='none', hoverinfo='label+percent',
                                 domain=dict(x=[0.78,0.92], y=[0.90,0.97])))
        except Exception:
            pass

        # Percentage annotations — clearer: bold, slightly larger, contrasting bg and border
        long_pct = int(round(ls_ratio * 100.0))
        short_pct = int(round((1 - ls_ratio) * 100.0))
        ann_y = (pill_y0 + pill_y1) / 2.0  # centered vertically on the pill

        fig.add_annotation(x=inner_start + 0.01,
                           y=ann_y,
                           xref='paper',
                           yref='paper',
                           text=f"<b style='color:#e9fff0'>LONG {long_pct}%</b>",
                           showarrow=False,
                           xanchor='left',
                           font=dict(color='#e9fff0', size=12, family='Helvetica', weight='bold'),
                           ay=-15,
                           bgcolor='rgba(0,0,0,0.45)',
                           bordercolor='rgba(255,255,255,0.06)',
                           borderwidth=1,
                           opacity=0.98)

        fig.add_annotation(x=inner_end - 0.01,
                           y=ann_y,
                           xref='paper',
                           yref='paper',
                           text=f"<b style='color:#ffecec'>SHORT {short_pct}%</b>",
                           showarrow=False,
                           xanchor='right',
                           font=dict(color='#ffecec', size=12, family='Helvetica', weight='bold'),
                           ay=-15,
                           bgcolor='rgba(0,0,0,0.45)',
                           bordercolor='rgba(255,255,255,0.06)',
                           borderwidth=1,
                           opacity=0.98)

    except Exception:
        pass

        # subtle border
        fig.add_shape(type='rect', xref='paper', yref='paper', x0=gx0, x1=gx1, y0=0.90, y1=0.98,
                      fillcolor='rgba(0,0,0,0)', line={'color':'rgba(255,255,255,0.14)','width':1.6}, layer='above')
        # percent texts
        fig.add_annotation(x=inner_start,
                           y=0.983,
                           xref='paper',
                           yref='paper',
                           text=f"<b style='color:#b8f7b8'>Long {ls_ratio*100:.0f}%</b>",
                           showarrow=False,
                           xanchor='left',
                           font=dict(color='#b8f7b8', size=11),
                           ay=-15)
        fig.add_annotation(x=inner_end,
                           y=0.983,
                           xref='paper',
                           yref='paper',
                           text=f"<b style='color:#ffb3b3'>Short {100-ls_ratio*100:.0f}%</b>",
                           showarrow=False,
                           xanchor='right',
                           font=dict(color='#ffb3b3', size=11),
                           ay=-15)
    except Exception:
        pass

    # OHLC
    try:
        fig.add_trace(go.Ohlc(
            x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c'],
            increasing_line_color='green', decreasing_line_color='red', name='OHLC'
        ))
    except Exception:
        pass

    # (retain existing heatbar / shapes / indicators — reproduce as in original)
    try:
        high, low, close_p = df['h'].iloc[-1], df['l'].iloc[-1], df['c'].iloc[-1]
        pivot = (high + low + close_p) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        levels = [('R2', r2), ('R1', r1), ('Pivot', pivot), ('S1', s1), ('S2', s2)]
        max_dist = max(abs(price - r2), abs(price - s2), 1e-6)
        for name, lvl in levels:
            dist = abs(price - lvl)
            width = dist / max_dist * 0.3 + 0.02
            fig.add_shape(type='rect', xref='paper', yref='y',
                          x0=0, x1=width,
                          y0=lvl - price * 0.001, y1=lvl + price * 0.001,
                          fillcolor=('rgba(0,200,0,0.16)' if lvl <= price else 'rgba(200,0,0,0.16)'),
                          opacity=0.6, layer='below')
            fig.add_annotation(x=width + 0.01,
                               y=lvl,
                               xref='paper',
                               yref='y',
                               text=name,
                               showarrow=False,
                               font=dict(color='white', size=10),
                               ay=-15,
                               xanchor='left',
                               yanchor='middle')
    except Exception:
        pass

    # Current price marker & dashed line with right-hand price label
    try:
                # horizontal dashed price line across the chart area
        fig.add_shape(type='line', xref='paper', x0=0, x1=1,
                      yref='y', y0=price, y1=price,
                      line=dict(color='white', width=1, dash='dash'),
                      layer='above')
        # right-side price pill (paper x=1)
        price_display = format_price_for_display(price)
        fig.add_annotation(x=1,
                           y=price,
                           xref='paper',
                           yref='y',
                           text=f"<b>{price_display}</b>",
                           showarrow=False,
                           font=dict(color='white', size=11, family='Helvetica'),
                           ay=0,
                           bgcolor='rgba(0,0,0,0.6)',
                           bordercolor='rgba(255,255,255,0.12)',
                           borderwidth=1,
                           xanchor='left',
                           yanchor='middle',
                           ax=6)
    except Exception:
        pass

    # Trade markers removed as requested by user
    # ---------- LAST-CANDLE COLORED BLOCK MARKER (annotation above with arrow to avoid overlap) ----------
    try:
        open_tr = next((t for t in reversed(wallet.get('trades', [])) if t.get('pnl_scaled') is None), None)
        pos_amt = float(wallet.get('position', 0.0))
        if open_tr and pos_amt != 0:
            entry_price = float(open_tr.get('price') or open_tr.get('entry_price') or price)
            actual_pnl = (price - entry_price) * pos_amt
            notional = float(open_tr.get('notional_usd') or (abs(pos_amt) * entry_price))
            scale = float(settings.base_usd) / max(notional, 1e-8)
            pnl_scaled = actual_pnl * scale
            pos_type = "Long" if pos_amt > 0 else "Short"
        else:
            pos_type = "Neutral"
            pnl_scaled = 0.0

        pnl_display = f"{pnl_scaled:+.2f}"
        if pos_type == "Long":
            bgcolor = "rgba(0,160,80,0.95)"
            border = "rgba(0,220,120,1)"
            icon = "▲"
        elif pos_type == "Short":
            bgcolor = "rgba(200,30,30,0.95)"
            border = "rgba(255,80,80,1)"
            icon = "▼"
        else:
            bgcolor = "rgba(120,120,120,0.95)"
            border = "rgba(200,200,200,0.95)"
            icon = "•"

        # place annotation above the highest visible candle to avoid overlap
        top_price = float(df['h'].max())
        price_range = float(df['h'].max() - df['l'].min() if df['h'].max() != df['l'].min() else df['h'].max()*0.01)
        ann_y = top_price + price_range * 0.12

        # draw a subtle rectangular shape as background (xref paper can't easily be aligned to candle x,
        # instead use annotation with arrow so it sits above and points to last candle)
        fig.add_annotation(x=lt,
            y=ann_y,
            xref='x',
            yref='y',
            text=f"<b>{icon} {pos_type}</b><br>PnL: ${pnl_display}",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=border,
            ax=0,
            ay=-40,
            # arrow from annotation to the point (pixel offset)
            font=dict(color='white', size=12, family='Helvetica'),
            bgcolor=bgcolor,
            bordercolor=border,
            borderwidth=2,
            opacity=0.98,
            align='center')
    except Exception:
        log.debug("Last-candle colored marker failed", exc_info=True)
        log.debug("Last-candle colored marker failed", exc_info=True)

    # Auto adjust Y-axis
    fig.update_yaxes(autorange=True, automargin=True)

    # --- Remove old trade marker traces (triangle/x/circle) and add simplified open/close markers ---
    try:
        # Remove traces with marker symbols containing triangle, x, or circle
        filtered = []
        for tr in fig.data:
            try:
                m = getattr(tr, 'marker', None)
                sym = None
                if m is not None:
                    sym = getattr(m, 'symbol', None)
                # helper to detect bad symbols
                def _bad(symb):
                    if symb is None:
                        return False
                    if isinstance(symb, (list, tuple)):
                        for s in symb:
                            if isinstance(s, str) and ('triangle' in s or s == 'x' or 'circle' in s):
                                return True
                        return False
                    if isinstance(symb, str):
                        return ('triangle' in symb) or (symb == 'x') or ('circle' in symb)
                    return False
                if _bad(sym):
                    # skip old marker trace
                    continue
            except Exception:
                pass
            filtered.append(tr)
        # reassign filtered traces
        fig.data = tuple(filtered)
    except Exception:
        try:
            log.debug('Old marker cleanup failed', exc_info=True)
        except Exception:
            pass

    # Build and add simplified open/close markers on top
# Build and add simplified open/close markers on top (show only last trade)
    try:
        # show only the last trade to keep UI focused
        last_trade = None
        try:
            trades = wallet.get('trades', []) if wallet is not None else []
            if trades:
                last_trade = trades[-1]
        except Exception:
            last_trade = None

        if last_trade is not None:
            # determine if last trade is within visible window
            try:
                tt_raw = pd.to_datetime(last_trade.get('time'))
                visible = (ft <= tt_raw <= lt)
            except Exception:
                visible = True

            if visible:
                # prepare entry marker
                try:
                    entry_time = pd.to_datetime(last_trade.get('time'))
                    entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
                    entry_t = df.index[entry_idx]
                except Exception:
                    entry_t = lt
                entry_p = float(last_trade.get('price') or last_trade.get('entry_price') or price)
                side = (last_trade.get('type') or last_trade.get('side') or '').strip()
                open_color = 'green' if 'Long' in side or side.lower()=='long' else 'red'

                
                # visual notification for the trade (entry / duration / close)
                try:
                    # compute exit_t/exit_p as before (if present)
                    exit_t = None
                    try:
                        if last_trade.get('exit_time') and last_trade.get('exit_price') is not None:
                            try:
                                exit_time = pd.to_datetime(last_trade.get('exit_time'))
                                exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]
                                exit_t = df.index[exit_idx]
                            except Exception:
                                exit_t = None
                            if exit_t is not None:
                                exit_p = float(last_trade.get('exit_price'))
                            else:
                                exit_p = None
                        else:
                            exit_p = None
                    except Exception:
                        exit_t = None
                        exit_p = None
                    # call helper to draw annotations + shaded duration
                    try:
                        notify_trade(fig, entry_t, entry_p, side=side, exit_t=exit_t, exit_p=exit_p, trade_id=(last_trade.get('id') or last_trade.get('trade_id')))
                    except Exception:
                        pass
                except Exception:
                    pass
        # bottom-right footer annotation with last trade details; always only one trade shown
# bottom-right footer annotation with last trade details; always only one trade shown
                try:
                    try:
                        ot = pd.to_datetime(last_trade.get('time'))
                        ot_s = ot.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        ot_s = str(last_trade.get('time') or '-')
                    if last_trade.get('exit_time'):
                        try:
                            ct = pd.to_datetime(last_trade.get('exit_time'))
                            ct_s = ct.strftime('%Y-%m-%d %H:%M')
                        except Exception:
                            ct_s = str(last_trade.get('exit_time') or '-')
                    else:
                        ct_s = '-'
                    # compute progress: unrealized PnL if open, else realized scaled pnl
                    if last_trade.get('exit_price') is None:
                        try:
                            pos_amt = float(wallet.get('position', 0.0))
                            unreal = (price - entry_p) * pos_amt if pos_amt != 0 else 0.0
                            progress = f"Unreal PnL: {unreal:+.2f}"
                        except Exception:
                            progress = ''
                    else:
                        progress = f"PnL: {float(last_trade.get('pnl_scaled') or 0.0):+.2f}"

                    footer = f"Last trade — {side} • Open: {ot_s} @ {entry_p:.6f} • Close: {ct_s} • {progress}"
                    fig.add_annotation(x=0.01, y=0.01, xref='paper', yref='paper', text=footer,
                                       showarrow=False, font=dict(color='white', size=11),
                                       bgcolor='rgba(0,0,0,0.6)', bordercolor='rgba(255,255,255,0.12)', borderwidth=1,
                                       xanchor='left', yanchor='bottom')
                except Exception:
                    pass
    except Exception:
        try:
            log.debug('Simplified markers drawing failed', exc_info=True)
        except Exception:
            pass

    # --- end of simplified markers ---

    # ------------------ METRICS computation & table enhancement ------------------
    try:
        # closed trades and realized pnl
        closed_trades = [t for t in wallet.get('trades', []) if (t.get('exit_price') is not None or t.get('pnl_scaled') is not None)]
        realized_total = 0.0
        pnl_list_scaled = []
        wins = 0
        for t in closed_trades:
            try:
                if t.get('pnl_scaled') is None and t.get('exit_price') is not None:
                    typ = t.get('type','Long')
                    entry_price = float(t.get('price') or t.get('entry_price') or 0.0)
                    exit_price = float(t.get('exit_price'))
                    size_amt = float(t.get('size',0.0))
                    pnl_usdt = (exit_price - entry_price) * size_amt if typ=='Long' else (entry_price - exit_price) * size_amt
                    not_usd = float(t.get('notional_usd') or max(abs(size_amt) * entry_price,1.0))
                    t['pnl'] = float(pnl_usdt)
                    t['pnl_scaled'] = float(pnl_usdt * (settings.base_usd / max(not_usd,1e-8)))
                if t.get('pnl_scaled') is not None:
                    realized_total += float(t.get('pnl_scaled'))
                    pnl_list_scaled.append(float(t.get('pnl_scaled')))
                    if float(t.get('pnl_scaled')) > 0:
                        wins += 1
            except Exception:
                continue

        closed_count = len(pnl_list_scaled)
        avg_trade_pnl = float(np.mean(pnl_list_scaled)) if pnl_list_scaled else 0.0
        win_rate = (wins / closed_count * 100.0) if closed_count else 0.0
        # last closed trade pnl/time
        last_closed = next((t for t in reversed(wallet.get('trades', [])) if t.get('exit_price') is not None), None)
        last_trade_pnl = float(last_closed.get('pnl_scaled')) if last_closed and last_closed.get('pnl_scaled') is not None else 0.0
        last_exit_time = str(last_closed.get('exit_time')) if last_closed and last_closed.get('exit_time') is not None else ""

        # unrealized for open trade
        open_tr = next((t for t in reversed(wallet.get('trades', [])) if t.get('pnl_scaled') is None), None)
        if open_tr and wallet.get('position',0.0) != 0:
            try:
                actual_init = (price - float(open_tr.get('price'))) * wallet.get('position',0.0)
                not_open = float(open_tr.get('notional_usd') or (abs(open_tr.get('size',0.0)) * open_tr.get('price',1.0)))
                scale_init = float(settings.base_usd) / max(not_open, 1e-8)
                init_pnl = actual_init * scale_init
            except Exception:
                init_pnl = 0.0
            open_position = open_tr.get('type','Neutral')
            open_size = float(open_tr.get('size',0.0))
        else:
            init_pnl = 0.0
            open_position = "Neutral"
            open_size = 0.0

                # Merge server-side wallets into client 'store' so cumulative PnL includes historic symbols
        try:
            if isinstance(store, dict):
                try:
                    with state_lock:
                        server_wallets = active_state.get('wallets', {})
                        if server_wallets:
                            store.setdefault('wallets', {})
                            # overwrite or add server wallets into the client store representation
                            for s, w in server_wallets.items():
                                store['wallets'][s] = w
                except Exception:
                    pass
        except Exception:
            pass

        # Aggregate realized pnl across all symbols/wallets so table shows cumulative PnL
        try:
            aggregated_realized = 0.0
            if isinstance(store, dict) and store.get('wallets'):
                for w in store.get('wallets', {}).values():
                    for tt in w.get('trades', []):
                        try:
                            if tt.get('pnl_scaled') is not None:
                                aggregated_realized += float(tt.get('pnl_scaled') or 0.0)
                        except Exception:
                            continue
            else:
                aggregated_realized = realized_total
        except Exception:
            aggregated_realized = realized_total

        # Use aggregated_realized in metrics (cumulative across symbols)
        equity_val = int(round(settings.base_usd + realized_total + init_pnl))
        
        metrics = {
            "equity": equity_val,
            "realized_pnl": round(float(aggregated_realized), 2),
            "total_realized_pnl": round(float(aggregated_realized), 2),
            "unrealized_pnl": round(float(init_pnl), 2),
            "last_trade_pnl": round(float(last_trade_pnl), 2),
            "last_exit_time": last_exit_time,
            "open_position": open_position,
            "open_size": round(open_size, 6),
            "trades_count": int(closed_count),
            "win_rate": round(float(win_rate), 2),
            "avg_trade_pnl": round(float(avg_trade_pnl), 2),
            "liq_usd": int(round(adaptive.get('liquidity_usd_recent', 0.0))),
            "liq_score": float(round(adaptive.get('liquidity_score',0.0), 4)),
            "signal_strength": float(round(min(1.0, max(score_long, score_short)), 3)),
            "er": float(round(er, 4)),
            "ae_conf": float(round(ae_conf, 4))
        }
    except Exception:
        metrics = {
            "equity": int(round(settings.base_usd)),
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "last_trade_pnl": 0.0,
            "last_exit_time": "",
            "open_position": "Neutral",
            "open_size": 0.0,
            "trades_count": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "liq_usd": int(round(adaptive.get('liquidity_usd_recent', 0.0))),
            "liq_score": float(round(adaptive.get('liquidity_score',0.0), 4)),
            "signal_strength": 0.0
        }

    # 
    # --- Custom markers: current price circle, horizontal price label, closed-trade PnL markers, Fibonacci levels ---
    try:
        now_x = pd.Timestamp.now(tz='Europe/Kyiv') if 'pd' in globals() else datetime.datetime.now()
        cur_price = float(price) if 'price' in locals() or 'price' in globals() else float(wallet.get('last_price', 0))
        # determine color: based on last open trade type if there's an open position
        cur_col = 'gray'
        try:
            pos = float(wallet.get('position', 0.0))
        except Exception:
            pos = 0.0
        if pos > 0:
            cur_col = 'green'
        elif pos < 0:
            cur_col = 'red'
        else:
            # fallback: check most recent open trade if any
            try:
                for t in reversed(wallet.get('trades', [])):
                    if not t.get('exit_time') and not t.get('exit_price'):
                        cur_col = 'green' if 'long' in str(t.get('type','')).lower() else 'red'
                        break
            except Exception:
                pass

        # Circle marker at current local time + price (shows pnl for the active trade if available)
        try:
            # determine color by current wallet position (cur_col already set above)
            cur_size = 12
            cur_pnl_label = ''
            open_tr = None
            # find most recent open trade without exit
            try:
                for t in reversed(wallet.get('trades', [])):
                    if not t.get('exit_time') and not t.get('exit_price'):
                        open_tr = t
                        break
            except Exception:
                open_tr = None
            # compute unrealized pnl if open trade exists
            if open_tr is not None:
                try:
                    entry_price = float(open_tr.get('price') or open_tr.get('entry_price') or cur_price)
                    size_amt = float(open_tr.get('size') or 0.0)
                    if size_amt != 0:
                        if str(open_tr.get('type','')).lower().startswith('long'):
                            pnl_usdt = (cur_price - entry_price) * size_amt
                        else:
                            pnl_usdt = (entry_price - cur_price) * size_amt
                        not_usd = float(open_tr.get('notional_usd') or max(abs(size_amt) * entry_price, 1.0))
                        pnl_scaled = float(pnl_usdt * (settings.base_usd / max(not_usd, 1e-8))) if hasattr(settings, 'base_usd') else pnl_usdt
                        pct = (pnl_scaled / settings.base_usd * 100.0) if getattr(settings, 'base_usd', None) else 0.0
                        cur_pnl_label = f"{int(round(pnl_scaled)):+d} ({pct:+.2f}%)"
                except Exception:
                    cur_pnl_label = ''
            fig.add_trace(go.Scatter(
                x=[now_x], y=[cur_price], mode='markers+text',
                marker=dict(symbol='circle', size=cur_size, color=cur_col, line=dict(color='white', width=1)),
                text=[cur_pnl_label], textposition='top right', hoverinfo='text', showlegend=False
            ))
        except Exception:
            pass

        # Horizontal dashed price line across the visible candles + right-side price label
        try:
            x0 = df.index[0]; x1 = df.index[-1]
            fig.add_shape(type='line', x0=x0, x1=x1, y0=cur_price, y1=cur_price,
                          line=dict(color=cur_col, width=1, dash='dash'), xref='x', yref='y', layer='above')
        except Exception:
            pass

    # Closed trade markers removed as requested
        # Add Fibonacci horizontal lines based on recent swing high/low (simple static fibs)
        try:
            look = min(len(df), 100)
            if look >= 2:
                recent = df.tail(look)
                swing_high = float(recent['h'].max())
                swing_low = float(recent['l'].min())
                diff = swing_high - swing_low if swing_high > swing_low else 1e-6
                # Fibonacci levels and a small color palette (0%->100%)
                levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                colors = ['#7f7f7f', '#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c', '#17becf', '#d62728']
                for i, lvl in enumerate(levels):
                    yv = swing_high - diff * lvl
                    c = colors[i % len(colors)]
                    fig.add_shape(type='line', x0=recent.index[0], x1=recent.index[-1], y0=yv, y1=yv,
                                  line=dict(color=c, width=1, dash='dot'), xref='x', yref='y', layer='below')
                    fig.add_annotation(xref='paper', x=1.005, y=yv, xanchor='left', yanchor='middle',
                    text=f"Fib {int(lvl*100)}% {format_price_for_display(yv)}", showarrow=False,
                                       font=dict(color=c, size=10),
                                       bgcolor='rgba(0,0,0,0.25)')
        except Exception:
            pass
    except Exception:
        pass

    # serialize timestamps for dash store
    for t in wallet.get('trades', []):
        if isinstance(t.get('time'), pd.Timestamp):
            t['time'] = t['time'].isoformat()
        if t.get('exit_time') and isinstance(t['exit_time'], pd.Timestamp):
            t['exit_time'] = t['exit_time'].isoformat()

    # ------------------ METRICS computation & table enhancement (robust) ------------------
    try:
        # closed trades and realized pnl (safe accumulation)
        closed_trades = [t for t in wallet.get('trades', []) if (t.get('exit_price') is not None or t.get('pnl_scaled') is not None)]
        total_realized_pnl = 0.0
        pnl_list_scaled = []
        wins = 0
        for t in closed_trades:
            try:
                # compute pnl if missing but exit_price exists
                if t.get('pnl_scaled') is None and t.get('exit_price') is not None:
                    typ = t.get('type','Long')
                    entry_price = float(t.get('price') or t.get('entry_price') or 0.0)
                    exit_price = float(t.get('exit_price'))
                    size_amt = float(t.get('size',0.0))
                    pnl_usdt = (exit_price - entry_price) * size_amt if typ=='Long' else (entry_price - exit_price) * size_amt
                    not_usd = float(t.get('notional_usd') or max(abs(size_amt) * entry_price,1.0))
                    t['pnl'] = float(pnl_usdt)
                    t['pnl_scaled'] = float(pnl_usdt * (settings.base_usd / max(not_usd,1e-8)))
                if t.get('pnl_scaled') is not None:
                    total_realized_pnl += float(t.get('pnl_scaled'))
                    pnl_list_scaled.append(float(t.get('pnl_scaled')))
                    if float(t.get('pnl_scaled')) > 0:
                        wins += 1
            except Exception:
                # skip problematic trade record
                continue

        closed_count = len(pnl_list_scaled)
        avg_trade_pnl = float(np.mean(pnl_list_scaled)) if pnl_list_scaled else 0.0
        win_rate = (wins / closed_count * 100.0) if closed_count else 0.0

        # last closed trade info (safe)
        last_closed = next((t for t in reversed(wallet.get('trades', [])) if t.get('exit_price') is not None and t.get('pnl_scaled') is not None), None)
        last_trade_pnl = float(last_closed.get('pnl_scaled')) if last_closed else 0.0
        last_exit_time = str(last_closed.get('exit_time')) if last_closed and last_closed.get('exit_time') is not None else ""

        # unrealized for open trade (safe)
        open_tr = next((t for t in reversed(wallet.get('trades', [])) if t.get('pnl_scaled') is None), None)
        if open_tr and wallet.get('position',0.0) != 0:
            try:
                actual_init = (price - float(open_tr.get('price'))) * wallet.get('position',0.0)
                not_open = float(open_tr.get('notional_usd') or (abs(open_tr.get('size',0.0)) * open_tr.get('price',1.0)))
                scale_init = float(settings.base_usd) / max(not_open, 1e-8)
                init_pnl = actual_init * scale_init
            except Exception:
                init_pnl = 0.0
            open_position = open_tr.get('type','Neutral')
            open_size = float(open_tr.get('size',0.0))
        else:
            init_pnl = 0.0
            open_position = "Neutral"
            open_size = 0.0

        equity_val = int(round(settings.base_usd + total_realized_pnl + init_pnl))

        metrics = {
            "equity": equity_val,
            "realized_pnl": round(float(aggregated_realized), 2),
            "total_realized_pnl": round(float(aggregated_realized), 2),
            "unrealized_pnl": round(float(init_pnl), 2),
            "last_trade_pnl": round(float(last_trade_pnl), 2),
            "last_exit_time": last_exit_time,
            "open_position": open_position,
            "open_size": round(open_size, 6),
            "trades_count": int(closed_count),
            "win_rate": round(float(win_rate), 2),
            "avg_trade_pnl": round(float(avg_trade_pnl), 2),
            "liq_usd": int(round(adaptive.get('liquidity_usd_recent', 0.0))),
            "liq_score": float(round(adaptive.get('liquidity_score',0.0), 4)),
            "signal_strength": float(round(min(1.0, max(score_long, score_short)), 3)),
            "er": float(round(er, 4)),
            "ae_conf": float(round(ae_conf, 4))
        }
    except Exception:
        metrics = {
            "equity": int(round(settings.base_usd)),
            "realized_pnl": 0.0,
            "total_realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "last_trade_pnl": 0.0,
            "last_exit_time": "",
            "open_position": "Neutral",
            "open_size": 0.0,
            "trades_count": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "liq_usd": int(round(adaptive.get('liquidity_usd_recent', 0.0))),
            "liq_score": float(round(adaptive.get('liquidity_score',0.0), 4)),
            "signal_strength": 0.0
        }
        try:
            sanitize_annotations(fig)
        except Exception:
            pass

    try:
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', gridwidth=0.5)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', gridwidth=0.5)
    except Exception:
        pass

    cards = build_metrics_cards([metrics])
    return fig, cards, store

# ==================== 12. CLI ====================
cli = typer.Typer()

@cli.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        threading.Thread(target=lambda: asyncio.run(run_live()), daemon=True).start()
        app.run(host="0.0.0.0", port=8050, debug=False)

@cli.command()
def live():
    """Запустити тільки live-торгівлю"""
    asyncio.run(run_live())

@cli.command()
def chart():
    """Запустити тільки Dash-інтерфейс"""
    app.run(host="0.0.0.0", port=8050, debug=False)

if __name__ == "__main__":
    cli()

# Adaptive volatility helper (appended)
vol_state = {'ema': None, 'alpha': 0.12, 'hist': collections.deque(maxlen=2000)}
liq_state = {'hist': collections.deque(maxlen=2000)}

def safe_norm(x, lo, hi):
    try:
        x = float(x); lo = float(lo); hi = float(hi)
        if hi <= lo:
            return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    except Exception:
        return 0.5

def compute_adaptive_params(current_vol, liquidity_score=0.0, market_cap=None):
    try:
        vol_state['hist'].append(float(current_vol))
    except Exception:
        vol_state['hist'].append(0.0)
    if vol_state['ema'] is None:
        vol_state['ema'] = float(current_vol) if current_vol is not None else 0.0
    else:
        a = vol_state['alpha']
        vol_state['ema'] = a * (float(current_vol) if current_vol is not None else 0.0) + (1 - a) * vol_state['ema']
    vol_val = vol_state['ema'] or 0.0
    if len(vol_state['hist']) >= 20:
        lo = float(np.percentile(list(vol_state['hist']), 10))
        hi = float(np.percentile(list(vol_state['hist']), 90))
    else:
        lo = float(min(list(vol_state['hist'])) if vol_state['hist'] else vol_val * 0.5)
        hi = float(max(list(vol_state['hist'])) if vol_state['hist'] else vol_val * 1.5)
    vol_scale = safe_norm(vol_val, lo, hi)
    vol_factor = (1.0 - vol_scale)
    try:
        if market_cap is None or market_cap <= 0:
            mc_score = 0.5
        else:
            mc_lo, mc_hi = 1e7, 1e12
            mc_score = safe_norm(np.log10(market_cap), np.log10(mc_lo), np.log10(mc_hi))
    except Exception:
        mc_score = 0.5
    lw = getattr(settings, 'liquidity_weight', 0.6)
    mw = getattr(settings, 'marketcap_weight', 0.4)
    combined_liq = float(np.clip((liquidity_score * lw + mc_score * mw), 0.0, 1.0))
    score_thr = float(np.clip(
        settings.score_threshold * (1.0 - 0.45 * vol_factor) * (1.0 - 0.25 * combined_liq),
        0.03, 0.95))
    min_time = int(max(1, settings.min_time_between_trades * (1.0 - 0.5 * vol_factor)))
    atr_open = float(np.clip(settings.atr_multiplier_open * (1.0 - 0.6 * vol_factor), 0.01, 2.5))
    open_soft = float(np.clip(settings.open_softness_factor * (1.0 + 0.8 * vol_factor), 0.01, 1.5))
    min_ae = float(np.clip(settings.min_ae_confidence * (1.0 - 0.8 * vol_factor), 0.0, 0.6))
    kelly_cap = float(np.clip(0.08 * (1.0 + 0.7 * vol_factor) + 0.01 + 0.04 * combined_liq, 0.01, 0.30))
    stop_loss_atr = float(np.clip(settings.stop_loss_atr_mult * (1.0 - 0.25 * vol_factor), 0.2, 3.0))
    try:
        liq_state['hist'].append(float(liquidity_score))
    except Exception:
        pass
    return {
        'vol_scale': vol_scale,
        'vol_factor': vol_factor,
        'score_threshold': score_thr,
        'min_time_between_trades': min_time,
        'atr_multiplier_open': atr_open,
        'open_softness_factor': open_soft,
        'min_ae_confidence': min_ae,
        'kelly_cap': kelly_cap,
        'stop_loss_atr_mult': stop_loss_atr,
        'vol_ema': vol_val,
        'vol_lo': lo,
        'vol_hi': hi,
        'liquidity_score': combined_liq,
        'marketcap_score': mc_score,
        'market_cap': market_cap,
        'liquidity_usd_recent': (liq_state['hist'][-1] if liq_state['hist'] else 0.0)
    }

# --- Auto-switch on close: override execute_close_trade to switch symbols after close ---
try:
    _orig_execute_close_trade = globals().get('execute_close_trade')
except Exception:
    _orig_execute_close_trade = None

def _switch_to_next_symbol(trigger_sym=None):
    try:
        with state_lock:
            cur = trigger_sym or active_state.get('active_symbol')
            # prefer settings.symbols list if provided, otherwise use wallets keys
            sym_list = list(getattr(settings, 'symbols', [])) or list(active_state.get('wallets', {}).keys()) or [getattr(settings, 'symbol', None)]
            sym_list = [s for s in sym_list if s]
            # filter excluded bases
            try:
                sym_list = [s for s in sym_list if s.split('/')[0].upper() not in EXCLUDED_BASES]
            except Exception:
                pass
            sym_list = [s for s in sym_list if s]
            if not sym_list:
                return
            if cur in sym_list:
                try:
                    idx = sym_list.index(cur)
                    cand = sym_list[(idx + 1) % len(sym_list)]
                except Exception:
                    cand = sym_list[0]
            else:
                cand = sym_list[0]
            if cand and cand != cur:
                if 'wallets' not in active_state:
                    active_state['wallets'] = {}
                if cand not in active_state['wallets']:
                    active_state['wallets'][cand] = {'position': 0.0, 'trades': []}
                active_state['active_symbol'] = cand
                try:
                    log.info(f"Auto-switch: switched active_symbol from {cur} -> {cand} after close")
                except Exception:
                    pass
    except Exception:
        pass

def execute_close_trade(sym, wallet, tr, exit_price, df_for_feat=None, reason=None, dry=True):
    # call original close function and then auto-switch symbol if close succeeded
    res = None
    try:
        if _orig_execute_close_trade:
            res = _orig_execute_close_trade(sym, wallet, tr, exit_price, df_for_feat, reason, dry)
        else:
            res = None
    except Exception as e:
        try:
            log.exception("Wrapped execute_close_trade failed: %s", e)
        except Exception:
            pass
        res = None
    try:
        # if we have a valid close (pnl_scaled or exit_price set) trigger switch
        closed_flag = False
        if tr and (tr.get('exit_price') is not None or tr.get('pnl_scaled') is not None):
            closed_flag = True
        if res is not None or closed_flag:
            # small delay-safe guard: perform switch under state_lock
            try:
                _switch_to_next_symbol(sym)
            except Exception:
                pass
    except Exception:
        pass
    return res
