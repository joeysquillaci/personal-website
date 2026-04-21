import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import sys
import time
from datetime import datetime


from technical_helpers import (
    FEATURE_DISPLAY,
    calculate_macd,
    calculate_rsi,
    create_sequences,
    find_support_resistance,
)
from modeling import EarlyStopping, LSTMModel
from education_content import EDU_TOPICS
from view_helpers import (
    open_eval_chart_settings,
    open_last_session_detail,
    populate_education_cards,
    redraw_eval_chart,
    update_last_session_card,
)


# ═══════════════════════════════════════════════════════════════════════════
#  GUI  –  LSTM Time-Series Workbench
# ═══════════════════════════════════════════════════════════════════════════

class LSTMWorkbenchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Track 2 - LSTM Time-Series Model Training & Evaluation")
        self.root.geometry("1360x950")
        self.root.minsize(1200, 820)

        self.ticker_var   = tk.StringVar(value="SPY")
        self.period_var   = tk.StringVar(value="5 Years")
        self.epochs_var   = tk.IntVar(value=100)
        self.lr_var       = tk.DoubleVar(value=0.001)
        self.split_var    = tk.DoubleVar(value=0.70)
        self.loss_var     = tk.StringVar(value="Standard MSE")
        self.lookback_var = tk.IntVar(value=20)
        self.horizon_var  = tk.StringVar(value="Medium-term")
        self.status_var   = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self.period_options = {
            "3 Months": "3mo",  "6 Months": "6mo",
            "1 Year":   "1y",   "2 Years":  "2y",
            "5 Years":  "5y",   "10 Years": "10y",
            "Max":      "max",
        }
        # Preset bundles for common market horizons. Custom keeps manual control.
        self.horizon_presets = {
            "Short-term": {
                "period": "6 Months",
                "lookback": 10,
                "epochs": 70,
                "lr": 0.0015,
                "split": 0.70,
                "loss": "Huber Loss",
            },
            "Medium-term": {
                "period": "2 Years",
                "lookback": 20,
                "epochs": 100,
                "lr": 0.0010,
                "split": 0.75,
                "loss": "Huber Loss",
            },
            "Long-term": {
                "period": "5 Years",
                "lookback": 30,
                "epochs": 130,
                "lr": 0.0007,
                "split": 0.80,
                "loss": "Huber Loss",
            },
        }
        self.custom_config_visible = False
        self._education_loaded = False
        self._apply_horizon_preset(self.horizon_var.get())
        for var in (
            self.period_var,
            self.epochs_var,
            self.lr_var,
            self.split_var,
            self.loss_var,
            self.lookback_var,
        ):
            var.trace_add("write", lambda *_: self._sync_horizon_ui())
        self._running = False
        self._link_counter = 0
        self.eval_chart_tf_var = tk.StringVar(value="6mo")
        self.eval_show_supports_var = tk.BooleanVar(value=False)
        self.eval_show_resistances_var = tk.BooleanVar(value=False)
        self.eval_sr_levels_var = tk.IntVar(value=2)
        self._eval_chart_payload = None
        self._eval_tf_buttons = {}
        self._eval_settings_btn = None
        self._eval_sr_cache = {}
        self._eval_content_cfg_after = None
        self._eval_canvas_cfg_after = None
        self._last_report_height_key = None
        self._last_session = None
        self._last_session_pending_show = False
        self._eval_wheel_remainder = 0.0
        self._eval_settings_path = os.path.join(
            os.path.dirname(__file__), ".lstm_eval_chart_settings.json"
        )
        self._load_eval_chart_settings()
        self._build_ui()
        self._last_log_flush_t = 0.0
        self._last_console_flush_t = 0.0
        self._console_max_lines = 600

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Notebook with three tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_training = ttk.Frame(self.notebook)
        self.tab_eval     = ttk.Frame(self.notebook)
        self.tab_edu      = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_training, text="Training Dashboard")
        self.notebook.add(self.tab_eval,     text="Evaluation")
        self.notebook.add(self.tab_edu,      text="Education")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ── Training Dashboard tab ─────────────────────────────────────
        training_shell = ttk.Frame(self.tab_training, padding=10)
        training_shell.pack(fill=tk.BOTH, expand=True)

        welcome_header = ttk.Frame(training_shell)
        welcome_header.pack(fill=tk.X, pady=(4, 10))
        ttk.Label(
            welcome_header,
            text="Welcome to Equity Education!",
            font=("Helvetica", 18, "bold"),
            justify="center",
            anchor="center",
        ).pack(fill=tk.X)
        ttk.Label(
            welcome_header,
            text="Begin learning by configuring the model below",
            font=("Helvetica", 13),
            justify="center",
            anchor="center",
        ).pack(fill=tk.X, pady=(2, 0))

        config_card = ttk.LabelFrame(
            training_shell, text="Configuration", padding=10
        )
        config_card.pack(fill=tk.X, pady=(0, 10))

        row_top = ttk.Frame(config_card)
        row_top.pack(anchor="center", pady=(0, 6))

        ttk.Label(row_top, text="Ticker").grid(row=0, column=0, padx=(0, 6))
        entry = ttk.Entry(row_top, textvariable=self.ticker_var, width=10)
        entry.grid(row=0, column=1, padx=(0, 14))
        entry.bind("<Return>", lambda e: self._on_submit())

        ttk.Label(row_top, text="Time Horizon").grid(row=0, column=2, padx=(0, 6))
        self.horizon_combo = ttk.Combobox(
            row_top,
            textvariable=self.horizon_var,
            values=["Short-term", "Medium-term", "Long-term", "Custom"],
            state="readonly",
            width=14,
        )
        self.horizon_combo.grid(row=0, column=3, padx=(0, 6))
        self.horizon_combo.bind("<<ComboboxSelected>>", self._on_horizon_changed)

        self.custom_config_btn = ttk.Button(
            row_top,
            text="⚙",
            width=3,
            command=self._toggle_custom_config,
        )
        self.custom_config_btn.grid(row=0, column=4, padx=(0, 8))
        self.custom_config_btn.grid_remove()

        self.preset_summary_label = ttk.Label(
            config_card,
            text="",
            justify="center",
        )
        self.preset_summary_label.pack(anchor="center", pady=(2, 10))

        self.custom_config_frame = ttk.LabelFrame(
            config_card,
            text="Custom Configuration",
            padding=8,
        )

        row_custom_top = ttk.Frame(self.custom_config_frame)
        row_custom_top.pack(anchor="center", pady=(0, 6))
        ttk.Label(row_custom_top, text="Data Period").grid(row=0, column=0, padx=(0, 6))
        self.period_combo = ttk.Combobox(
            row_custom_top,
            textvariable=self.period_var,
            values=list(self.period_options.keys()),
            state="readonly",
            width=10,
        )
        self.period_combo.grid(row=0, column=1, padx=(0, 14))

        ttk.Label(row_custom_top, text="Loss").grid(row=0, column=2, padx=(0, 6))
        self.loss_combo = ttk.Combobox(
            row_custom_top,
            textvariable=self.loss_var,
            values=["Standard MSE", "Huber Loss"],
            state="readonly",
            width=14,
        )
        self.loss_combo.grid(row=0, column=3, padx=(0, 12))

        row_custom_bottom = ttk.Frame(self.custom_config_frame)
        row_custom_bottom.pack(anchor="center")
        ttk.Label(row_custom_bottom, text="Epochs").grid(row=0, column=0, padx=(0, 6))
        self.epochs_entry = ttk.Entry(row_custom_bottom, textvariable=self.epochs_var, width=7)
        self.epochs_entry.grid(row=0, column=1, padx=(0, 14))
        ttk.Label(row_custom_bottom, text="Learning Rate").grid(
            row=0, column=2, padx=(0, 6)
        )
        self.lr_entry = ttk.Entry(row_custom_bottom, textvariable=self.lr_var, width=8)
        self.lr_entry.grid(row=0, column=3, padx=(0, 14))
        ttk.Label(row_custom_bottom, text="Train Split").grid(row=0, column=4, padx=(0, 6))
        self.split_entry = ttk.Entry(row_custom_bottom, textvariable=self.split_var, width=7)
        self.split_entry.grid(row=0, column=5, padx=(0, 14))
        ttk.Label(row_custom_bottom, text="Lookback").grid(row=0, column=6, padx=(0, 6))
        self.lookback_entry = ttk.Entry(row_custom_bottom, textvariable=self.lookback_var, width=7)
        self.lookback_entry.grid(row=0, column=7)

        # Keep advanced controls hidden until user selects Custom and opens submenu.
        self._sync_horizon_ui(force=True)

        row_submit = ttk.Frame(config_card)
        row_submit.pack(anchor="center", pady=(8, 0))
        self.submit_btn = ttk.Button(
            row_submit, text="Train and Evaluate", command=self._on_submit
        )
        self.submit_btn.pack()

        self.training_live_row = ttk.Frame(training_shell)
        self.training_live_row.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.training_live_row.columnconfigure(0, weight=1)
        self.training_live_row.columnconfigure(1, weight=1)

        self.console_card = ttk.LabelFrame(self.training_live_row, text="Training Console", padding=8)
        self.console_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.console_listbox = tk.Listbox(
            self.console_card,
            height=12,
            font=("Courier", 14),
            activestyle="none",
        )
        console_sb = ttk.Scrollbar(
            self.console_card, orient=tk.VERTICAL, command=self.console_listbox.yview
        )
        self.console_listbox.configure(yscrollcommand=console_sb.set)
        console_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.train_chart_card = ttk.LabelFrame(
            self.training_live_row, text="Training Loss Graph", padding=8
        )
        self.train_chart_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.fig_train = Figure(figsize=(5.0, 3.4), dpi=100, tight_layout=True)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.train_chart_card)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.training_live_row.pack_forget()

        self.last_session_card = ttk.LabelFrame(
            training_shell, text="Last Training Session", padding=10
        )
        self.last_session_card.pack(fill=tk.X, pady=(0, 10))
        self.last_session_title = ttk.Label(
            self.last_session_card,
            text="No completed session yet.",
            font=("Helvetica", 11, "bold"),
            justify="center",
        )
        self.last_session_title.pack(anchor="center")
        self.last_session_meta = ttk.Label(
            self.last_session_card,
            text=(
                "Run training to populate this snapshot.\n"
                "This panel shows key outcomes from the latest completed run."
            ),
            justify="center",
        )
        self.last_session_meta.pack(anchor="center", pady=(6, 0))
        self.last_session_hint = ttk.Label(
            self.last_session_card,
            text="Click to open detailed session view (console + loss chart).",
            foreground="#1976d2",
        )
        self.last_session_hint.pack(anchor="center", pady=(8, 0))

        interactive_widgets = (
            self.last_session_card,
            self.last_session_title,
            self.last_session_meta,
            self.last_session_hint,
        )
        for w in interactive_widgets:
            w.bind("<Button-1>", lambda e: self._open_last_session_detail())
            try:
                w.configure(cursor="hand2")
            except Exception:
                pass
        self.last_session_card.pack_forget()

        self.status_card = ttk.LabelFrame(training_shell, text="Run Status", padding=8)
        self.status_card.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(self.status_card, textvariable=self.status_var).pack(anchor="w", pady=(0, 6))
        self.progress = ttk.Progressbar(
            self.status_card, variable=self.progress_var, mode="determinate"
        )
        self.progress.pack(fill=tk.X)
        self.status_card.pack_forget()

        # ── Evaluation tab (scrollable container, static report box) ───
        self.eval_canvas = tk.Canvas(self.tab_eval, highlightthickness=0)
        self.eval_scrollbar = ttk.Scrollbar(
            self.tab_eval, orient=tk.VERTICAL, command=self.eval_canvas.yview
        )
        self.eval_canvas.configure(yscrollcommand=self.eval_scrollbar.set)
        self.eval_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.eval_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.eval_content = ttk.Frame(self.eval_canvas)
        self.eval_canvas_window = self.eval_canvas.create_window(
            (0, 0), window=self.eval_content, anchor="nw"
        )
        self.eval_content.bind("<Configure>", self._on_eval_content_configure)
        self.eval_canvas.bind("<Configure>", self._on_eval_canvas_configure)
        # Bind smooth wheel handling for evaluation scroll.
        self._bind_eval_mousewheel()

        # Placeholder encouraging user to train first
        self.eval_placeholder = ttk.Frame(self.eval_content, padding=10)
        self.eval_placeholder.pack(fill=tk.X, padx=10, pady=(10, 4))
        msg = ttk.Label(
            self.eval_placeholder,
            text=("No evaluation results yet.\n"
                  "Go to the Training Dashboard to configure and train the model."),
            justify="left"
        )
        msg.pack(side=tk.LEFT)
        go_btn = ttk.Button(
            self.eval_placeholder,
            text="Go to Training Dashboard",
            command=lambda: self.notebook.select(self.tab_training),
        )
        go_btn.pack(side=tk.RIGHT, padx=8)

        self.eval_loading_frame = ttk.Frame(self.eval_content, padding=(10, 2))
        self.eval_loading_label = ttk.Label(
            self.eval_loading_frame,
            text="Loading evaluation content...",
        )
        self.eval_loading_label.pack(side=tk.LEFT)
        self.eval_loading_bar = ttk.Progressbar(
            self.eval_loading_frame,
            mode="indeterminate",
            length=220,
        )
        self.eval_loading_bar.pack(side=tk.LEFT, padx=(10, 0))
        self.eval_loading_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.eval_loading_frame.pack_forget()

        # Summary cards for market movement and model outlook
        header_cards = ttk.Frame(self.eval_content, padding=8)
        header_cards.pack(fill=tk.X, padx=10, pady=(0, 2))
        header_cards.columnconfigure(0, weight=1)
        header_cards.columnconfigure(1, weight=1)
        header_cards.columnconfigure(2, weight=1)

        prev_card = ttk.LabelFrame(header_cards, text="", padding=8)
        prev_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.eval_prev_price_label = ttk.Label(
            prev_card, text="Prev Close: -", justify="center")
        self.eval_prev_price_label.pack(anchor="center")
        self.eval_prev_move_label = ttk.Label(
            prev_card, text="Change: -", font=("Helvetica", 12, "bold"), justify="center")
        self.eval_prev_move_label.pack(anchor="center", pady=(2, 0))

        current_card = ttk.LabelFrame(header_cards, text="", padding=8)
        current_card.grid(row=0, column=1, sticky="nsew", padx=3)
        self.eval_current_price_label = ttk.Label(
            current_card,
            text="$-",
            font=("Helvetica", 20, "bold"),
            justify="center")
        self.eval_current_price_label.pack(anchor="center", pady=(2, 2))
        self.eval_current_meta_label = ttk.Label(
            current_card, text=self.ticker_var.get().strip().upper(), justify="center")
        self.eval_current_meta_label.pack(anchor="center", pady=(0, 2))

        outlook_card = ttk.LabelFrame(header_cards, text="", padding=8)
        outlook_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.eval_pred_move_label = ttk.Label(
            outlook_card,
            text="Next-Day Move: -",
            font=("Helvetica", 12, "bold"),
            justify="center",
        )
        self.eval_pred_move_label.pack(anchor="center")
        self.eval_accuracy_label = ttk.Label(
            outlook_card, text="Directional Accuracy: -", justify="center"
        )
        self.eval_accuracy_label.pack(anchor="center", pady=(4, 0))

        # Evaluation chart container (graph + controls)
        chart_container = ttk.LabelFrame(self.eval_content, text="", padding=8)
        chart_container.pack(fill=tk.BOTH, expand=False, padx=10, pady=(2, 6))

        eval_chart_frame = ttk.Frame(chart_container, padding=2)
        # Keep chart area bounded so report + buttons remain on-screen.
        eval_chart_frame.pack(fill=tk.BOTH, expand=False, padx=2, pady=(2, 4))
        self.fig_eval = Figure(figsize=(9.0, 6.2), dpi=100, tight_layout=True)
        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, master=eval_chart_frame)
        self.canvas_eval.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        self.canvas_eval.get_tk_widget().configure(height=430)

        # Timeframe controls for evaluation chart
        tf_frame = ttk.Frame(chart_container, padding=(2, 0, 2, 2))
        tf_frame.pack(fill=tk.X)
        tf_frame.columnconfigure(0, weight=1)

        tf_left = ttk.Frame(tf_frame)
        tf_left.grid(row=0, column=0, sticky="w")

        tf_defs = [("1mo", "1M"), ("3mo", "3M"), ("6mo", "6M"), ("1y", "1Y")]
        for tf_key, tf_label in tf_defs:
            btn = ttk.Button(
                tf_left,
                text=tf_label,
                command=lambda k=tf_key: self._set_eval_chart_timeframe(k),
                state="disabled",
                width=6,
            )
            btn.pack(side=tk.LEFT, padx=3)
            self._eval_tf_buttons[tf_key] = btn
        self._eval_settings_btn = ttk.Button(
            tf_frame,
            text="⚙",
            width=4,
            state="disabled",
            command=self._open_eval_chart_settings,
        )
        self._eval_settings_btn.grid(row=0, column=1, sticky="e")

        log_frame = ttk.LabelFrame(self.eval_content, text="Model Evaluation Report")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 8))

        # Keep report text and detail buttons in the same section so
        # buttons are always visible directly below the report.
        report_body = ttk.Frame(log_frame)
        report_body.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        self.log_text = tk.Text(
            report_body, wrap=tk.WORD, font=("Courier", 11), height=30
        )
        # Non-scrollable report box; use Evaluation page scrolling instead.
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Detail buttons always shown below report text
        self.eval_details_frame = ttk.Frame(log_frame, padding=8)
        self.btn_training_config = ttk.Button(
            self.eval_details_frame,
            text="Training Configuration",
            command=lambda: self._show_eval_detail("training_config"),
        )
        self.btn_model_scorecard = ttk.Button(
            self.eval_details_frame,
            text="Model Scorecard",
            command=lambda: self._show_eval_detail("model_scorecard"),
        )
        self.btn_generalization_details = ttk.Button(
            self.eval_details_frame,
            text="Generalization Gap Details",
            command=lambda: self._show_eval_detail(
                "generalization_details"
            ),
        )
        # Lay out submenu buttons side-by-side.
        self.eval_details_frame.columnconfigure(0, weight=1)
        self.eval_details_frame.columnconfigure(1, weight=1)
        self.eval_details_frame.columnconfigure(2, weight=1)
        self.btn_training_config.grid(
            row=0, column=0, sticky="ew", padx=(0, 6), pady=0
        )
        self.btn_model_scorecard.grid(
            row=0, column=1, sticky="ew", padx=3, pady=0
        )
        self.btn_generalization_details.grid(
            row=0, column=2, sticky="ew", padx=(6, 0), pady=0
        )
        self.eval_details_frame.pack(fill=tk.X, padx=4, pady=(6, 8))

        # ── Education tab ──────────────────────────────────────────────
        edu_shell = ttk.Frame(self.tab_edu, padding=12)
        edu_shell.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            edu_shell,
            text="Education",
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            edu_shell,
            text=(
                "Click any card to open detailed guidance."
            ),
            font=("Helvetica", 11),
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.edu_loading_frame = ttk.Frame(edu_shell, padding=(0, 0, 0, 8))
        self.edu_loading_label = ttk.Label(
            self.edu_loading_frame,
            text="Loading education content...",
        )
        self.edu_loading_label.pack(side=tk.LEFT)
        self.edu_loading_bar = ttk.Progressbar(
            self.edu_loading_frame,
            mode="indeterminate",
            length=220,
        )
        self.edu_loading_bar.pack(side=tk.LEFT, padx=(10, 0))
        self.edu_loading_frame.pack(fill=tk.X)
        self.edu_loading_frame.pack_forget()

        self._edu_topics = EDU_TOPICS

        self.edu_cards = ttk.Frame(edu_shell)
        self.edu_cards.pack(fill=tk.BOTH, expand=True)

        for tag, color in [("green", "#00c853"), ("red", "#ef5350"),
                           ("yellow", "#ffca28"), ("orange", "#ff9800"),
                           ("cyan", "#00bcd4"), ("bold", None)]:
            if tag == "bold":
                self.log_text.tag_configure(
                    tag, font=("Courier", 11, "bold"))
            else:
                self.log_text.tag_configure(tag, foreground=color)
        self.log_text.tag_configure("summary", font=("Courier", 12, "bold"))

        # Link styling for clickable "here" text in the report
        self.log_text.tag_configure("link", foreground="#1976d2", underline=1)

    # ------------------------------------------------------------------ log
    def _log(self, msg, tag=None):
        line = f"{msg}\n"
        if tag:
            self.log_text.insert(tk.END, line, tag)
        else:
            self.log_text.insert(tk.END, line)
        self._trim_text_widget(self.log_text, max_lines=1500)
        now = time.monotonic()
        if now - self._last_log_flush_t >= 0.08:
            self.log_text.see(tk.END)
            self._last_log_flush_t = now

    def _section(self, title):
        self._log("")
        self._log(f"── {title} ──", "bold")

    def _autosize_eval_report_text(self):
        """Resize report text widget to fit all content (no inner scrolling)."""
        if not hasattr(self, "log_text"):
            return
        try:
            # Keep this lightweight; oversized text widgets can slow tab switches.
            logical_lines = int(self.log_text.index("end-1c").split(".")[0])
            # Cap height so evaluation tab remains responsive when revisited.
            target_height = max(8, min(44, logical_lines + 1))
            width_key = int(self.log_text.winfo_width())
            new_key = (logical_lines, width_key, target_height)
            if new_key == self._last_report_height_key:
                return
            self._last_report_height_key = new_key
            self.log_text.configure(height=target_height)
        except Exception:
            pass

    def _on_eval_content_configure(self, _event=None):
        """Debounce expensive eval scrollregion recomputations."""
        if self._eval_content_cfg_after is not None:
            try:
                self.root.after_cancel(self._eval_content_cfg_after)
            except Exception:
                pass
        self._eval_content_cfg_after = self.root.after(
            35, self._apply_eval_scrollregion
        )

    def _apply_eval_scrollregion(self):
        self._eval_content_cfg_after = None
        if not hasattr(self, "eval_canvas"):
            return
        try:
            self.eval_canvas.configure(scrollregion=self.eval_canvas.bbox("all"))
        except Exception:
            pass

    def _on_eval_canvas_configure(self, event):
        """Debounce canvas window width sync during resizes/tab changes."""
        width = int(event.width) if event is not None else None
        if self._eval_canvas_cfg_after is not None:
            try:
                self.root.after_cancel(self._eval_canvas_cfg_after)
            except Exception:
                pass
        self._eval_canvas_cfg_after = self.root.after(
            35, lambda w=width: self._apply_eval_canvas_width(w)
        )

    def _apply_eval_canvas_width(self, width):
        self._eval_canvas_cfg_after = None
        if width is None or not hasattr(self, "eval_canvas"):
            return
        try:
            self.eval_canvas.itemconfigure(self.eval_canvas_window, width=width)
        except Exception:
            pass

    def _bind_eval_mousewheel(self):
        """Bind global wheel events for evaluation scrolling."""
        self.root.bind_all("<MouseWheel>", self._on_eval_mousewheel)
        self.root.bind_all("<Button-4>", self._on_eval_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_eval_mousewheel_linux)

    def _unbind_eval_mousewheel(self):
        """Unbind global wheel events when pointer leaves evaluation area."""
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _is_widget_in_eval_area(self, widget):
        """Return True if widget is inside evaluation tab/canvas hierarchy."""
        cur = widget
        while cur is not None:
            if cur == self.eval_canvas or cur == self.eval_content or cur == self.tab_eval:
                return True
            cur = getattr(cur, "master", None)
        return False

    def _on_eval_mousewheel(self, event):
        """Trackpad/mouse-wheel scroll for evaluation canvas."""
        if not hasattr(self, "notebook") or not hasattr(self, "tab_eval"):
            return
        if self.notebook.select() != str(self.tab_eval):
            return
        target = self.root.winfo_containing(event.x_root, event.y_root)
        if target is not None and not self._is_widget_in_eval_area(target):
            return
        if event.delta == 0:
            return
        # Smooth wheel handling with delta accumulation.
        if sys.platform == "darwin":
            self._eval_wheel_remainder += (-event.delta) * 0.35
            steps = int(self._eval_wheel_remainder)
            if steps != 0:
                self.eval_canvas.yview_scroll(steps, "units")
                self._eval_wheel_remainder -= steps
        else:
            self._eval_wheel_remainder += (-event.delta / 120.0)
            steps = int(self._eval_wheel_remainder)
            if steps != 0:
                self.eval_canvas.yview_scroll(steps, "units")
                self._eval_wheel_remainder -= steps

    def _on_eval_mousewheel_linux(self, event):
        """Linux wheel fallback (<Button-4>/<Button-5>)."""
        if not hasattr(self, "notebook") or not hasattr(self, "tab_eval"):
            return
        if self.notebook.select() != str(self.tab_eval):
            return
        target = self.root.winfo_containing(event.x_root, event.y_root)
        if target is not None and not self._is_widget_in_eval_area(target):
            return
        if event.num == 4:
            self.eval_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.eval_canvas.yview_scroll(1, "units")

    def _set_eval_chart_timeframe(self, timeframe_key):
        """Update chart timeframe and redraw evaluation chart."""
        self.eval_chart_tf_var.set(timeframe_key)
        # Schedule on UI thread after layout settles.
        self.root.after_idle(self._refresh_eval_chart_if_ready)

    def _set_eval_tf_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in self._eval_tf_buttons.values():
            btn.configure(state=state)
        if self._eval_settings_btn is not None:
            self._eval_settings_btn.configure(state=state)

    def _load_eval_chart_settings(self):
        """Load persisted evaluation chart settings."""
        try:
            if not os.path.exists(self._eval_settings_path):
                return
            with open(self._eval_settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            legacy_show_sr = bool(data.get("show_support_resistance", False))
            self.eval_show_supports_var.set(
                bool(data.get("show_supports", legacy_show_sr))
            )
            self.eval_show_resistances_var.set(
                bool(data.get("show_resistances", legacy_show_sr))
            )
            levels = int(data.get("sr_levels", 2))
            self.eval_sr_levels_var.set(max(1, min(3, levels)))
        except Exception:
            # Ignore malformed/missing settings and keep defaults.
            pass

    def _save_eval_chart_settings(self):
        """Persist evaluation chart settings to disk."""
        try:
            show_supports = bool(self.eval_show_supports_var.get())
            show_resistances = bool(self.eval_show_resistances_var.get())
            data = {
                # Legacy key retained for backward compatibility.
                "show_support_resistance": bool(show_supports or show_resistances),
                "show_supports": show_supports,
                "show_resistances": show_resistances,
                "sr_levels": int(self.eval_sr_levels_var.get()),
            }
            with open(self._eval_settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _open_eval_chart_settings(self):
        open_eval_chart_settings(self)

    def _redraw_eval_chart(self):
        redraw_eval_chart(self)

    def _refresh_eval_chart_if_ready(self):
        """Best-effort chart refresh once payload/widget are ready."""
        if not self._eval_chart_payload:
            return
        if not hasattr(self, "canvas_eval"):
            return
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        try:
            self._redraw_eval_chart()
        except Exception:
            pass

    def _apply_eval_chart_payload(self, payload):
        """Store chart payload and redraw from the Tk main thread."""
        self._eval_chart_payload = payload
        self._eval_sr_cache.clear()
        self._set_eval_tf_buttons_enabled(True)
        # Draw immediately, then once more after geometry settles.
        self._refresh_eval_chart_if_ready()
        self.root.after(120, self._refresh_eval_chart_if_ready)

    def _popup_text(self, title, body_text, font_size=12):
        """Open a small read-only window with monospaced text."""
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("860x520")

        txt = tk.Text(top, wrap=tk.WORD, font=("Helvetica", font_size))
        sb = ttk.Scrollbar(top, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)

        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        txt.insert("1.0", body_text)
        txt.configure(state="disabled")

        # Keep focus on main window unless user clicks inside popup
        top.transient(self.root)
        top.grab_set()
        top.grab_release()

    def _show_eval_detail(self, which):
        """Show a popup for evaluation details from stored report text."""
        detail_map = {
            "training_config": (
                "Training Configuration",
                getattr(self, "_training_config_text", None),
            ),
            "model_scorecard": (
                "Model Scorecard",
                getattr(self, "_model_scorecard_text", None),
            ),
            "generalization_details": (
                "Generalization Gap Details",
                getattr(self, "_generalization_details_text", None),
            ),
        }

        if which not in detail_map:
            return

        title, body = detail_map[which]
        if not body:
            messagebox.showinfo(
                "Not ready yet",
                "Train the model first to view this evaluation detail.",
            )
            return

        self._popup_text(title, body)

    def _log_link(self, prefix, link_label, callback):
        """Insert a clickable 'link_label' that triggers callback."""
        # Prefix (plain)
        self.log_text.insert(tk.END, prefix)

        # Capture indices for the clickable label
        link_start = self.log_text.index(tk.END)
        self.log_text.insert(tk.END, link_label)
        link_end = self.log_text.index(tk.END)

        # Unique tag per link so each can bind to a different callback
        tag_name = f"link_{self._link_counter}"
        self._link_counter += 1
        self.log_text.tag_add(tag_name, link_start, link_end)
        # Make the clickable label visually obvious.
        self.log_text.tag_configure(
            tag_name,
            foreground="#1976d2",
            underline=1,
            font=("Courier", 11, "bold"),
        )
        self.log_text.tag_bind(
            tag_name, "<Button-1>", lambda e: callback())
        # Change the Text widget cursor only while hovering the clickable tag.
        # (Avoid passing `cursor=` into tag_configure: it's not supported everywhere.)
        self.log_text.tag_bind(
            tag_name, "<Enter>", lambda e: self.log_text.configure(cursor="hand2")
        )
        self.log_text.tag_bind(
            tag_name, "<Leave>", lambda e: self.log_text.configure(cursor="arrow")
        )

        # Newline after the line content
        self.log_text.insert(tk.END, "\n")
        now = time.monotonic()
        if now - self._last_log_flush_t >= 0.08:
            self.log_text.see(tk.END)
            self._last_log_flush_t = now

    def _console_log(self, msg):
        """Append a line to the live training console."""
        if not hasattr(self, "console_listbox"):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        self.console_listbox.insert(tk.END, line)
        self._trim_listbox_widget(self.console_listbox, max_rows=self._console_max_lines)
        now = time.monotonic()
        if now - self._last_console_flush_t >= 0.08:
            try:
                self.console_listbox.yview_moveto(1.0)
            except Exception:
                pass
            self._last_console_flush_t = now

    def _trim_text_widget(self, widget, max_lines):
        """Bound Text widget size so large logs don't slow tab rendering."""
        try:
            line_count = int(widget.index("end-1c").split(".")[0])
            overflow = line_count - int(max_lines)
            if overflow > 0:
                widget.delete("1.0", f"{overflow}.0")
        except Exception:
            pass

    def _trim_listbox_widget(self, widget, max_rows):
        """Bound Listbox rows to keep rendering fast."""
        try:
            size = int(widget.size())
            overflow = size - int(max_rows)
            if overflow > 0:
                widget.delete(0, overflow - 1)
        except Exception:
            pass

    def _update_last_session_card(self, session):
        update_last_session_card(self, session)

    def _open_last_session_detail(self):
        open_last_session_detail(self)

    def _set_status_pct(self, pct):
        """Show simplified progress-only status text."""
        pct_i = int(round(max(0.0, min(100.0, float(pct)))))
        self.status_var.set(f"{pct_i}% complete")

    def _set_eval_loading(self, is_loading):
        """Show or hide a loading indicator in the Evaluation tab."""
        if not hasattr(self, "eval_loading_frame") or not hasattr(self, "eval_loading_bar"):
            return
        try:
            if is_loading:
                if not self.eval_loading_frame.winfo_ismapped():
                    self.eval_loading_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
                self.eval_loading_bar.start(12)
            else:
                self.eval_loading_bar.stop()
                if self.eval_loading_frame.winfo_ismapped():
                    self.eval_loading_frame.pack_forget()
        except Exception:
            pass

    def _set_education_loading(self, is_loading):
        """Show or hide a loading indicator in the Education tab."""
        if not hasattr(self, "edu_loading_frame") or not hasattr(self, "edu_loading_bar"):
            return
        try:
            if is_loading:
                if not self.edu_loading_frame.winfo_ismapped():
                    self.edu_loading_frame.pack(fill=tk.X)
                self.edu_loading_bar.start(12)
            else:
                self.edu_loading_bar.stop()
                if self.edu_loading_frame.winfo_ismapped():
                    self.edu_loading_frame.pack_forget()
        except Exception:
            pass

    def _populate_education_cards(self):
        populate_education_cards(self)

    def _on_tab_changed(self, _event=None):
        """Populate/refresh tab content immediately on tab click."""
        try:
            selected = self.notebook.select()
        except Exception:
            return
        if selected == str(self.tab_edu):
            # Force immediate education-tab layout refresh on tab click.
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            if not self._education_loaded:
                self._set_education_loading(True)
                # Use near-immediate scheduling so render does not wait for hover/motion events.
                self.root.after(0, self._populate_education_cards)
            else:
                # Even when already loaded, nudge geometry recalculation for instant repaint.
                self.root.after_idle(self.root.update_idletasks)
        elif selected == str(self.tab_training):
            if (
                self._last_session is not None
                and self._last_session_pending_show
                and not self._running
                and hasattr(self, "last_session_card")
                and not self.last_session_card.winfo_ismapped()
            ):
                try:
                    if hasattr(self, "status_card"):
                        self.last_session_card.pack(
                            fill=tk.X, pady=(0, 10), before=self.status_card
                        )
                    else:
                        self.last_session_card.pack(fill=tk.X, pady=(0, 10))
                except Exception:
                    self.last_session_card.pack(fill=tk.X, pady=(0, 10))
                self._last_session_pending_show = False
            # Force immediate layout refresh on tab click (no hover dependency).
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            if hasattr(self, "canvas_train") and hasattr(self, "training_live_row"):
                try:
                    if self.training_live_row.winfo_ismapped():
                        self.canvas_train.draw_idle()
                except Exception:
                    pass
        elif selected == str(self.tab_eval):
            # Refresh evaluation layout/canvas immediately on tab click.
            self._on_eval_content_configure()
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            if self._eval_chart_payload:
                self._refresh_eval_chart_if_ready()

    def _apply_horizon_preset(self, horizon_name):
        cfg = self.horizon_presets.get(horizon_name)
        if not cfg:
            return
        self.period_var.set(cfg["period"])
        self.lookback_var.set(cfg["lookback"])
        self.epochs_var.set(cfg["epochs"])
        self.lr_var.set(cfg["lr"])
        self.split_var.set(cfg["split"])
        self.loss_var.set(cfg["loss"])

    def _preset_summary_text(self):
        return (
            f"Period {self.period_var.get()} | Lookback {self.lookback_var.get()}d | "
            f"Epochs {self.epochs_var.get()} | LR {self.lr_var.get():.4g} | "
            f"Train Split {self.split_var.get():.0%} | Loss {self.loss_var.get()}"
        )

    def _sync_horizon_ui(self, force=False):
        is_custom = self.horizon_var.get() == "Custom"
        if hasattr(self, "preset_summary_label"):
            self.preset_summary_label.config(text=self._preset_summary_text())

        if not hasattr(self, "custom_config_btn"):
            return

        if is_custom:
            # Show settings icon only for manual custom mode.
            self.custom_config_btn.grid()
            self.custom_config_btn.state(["!disabled"])
            if force and not self.custom_config_visible and hasattr(self, "custom_config_frame"):
                self.custom_config_frame.pack(fill=tk.X, pady=(0, 8))
                self.custom_config_visible = True
        else:
            # Hide settings icon for all preset horizons.
            self.custom_config_btn.grid_remove()
            self.custom_config_btn.state(["disabled"])
            if self.custom_config_visible and hasattr(self, "custom_config_frame"):
                self.custom_config_frame.pack_forget()
                self.custom_config_visible = False

    def _on_horizon_changed(self, _event=None):
        horizon_name = self.horizon_var.get()
        if horizon_name != "Custom":
            self._apply_horizon_preset(horizon_name)
        self._sync_horizon_ui()

    def _toggle_custom_config(self):
        if self.horizon_var.get() != "Custom":
            return
        if not hasattr(self, "custom_config_frame"):
            return
        if self.custom_config_visible:
            self.custom_config_frame.pack_forget()
            self.custom_config_visible = False
        else:
            self.custom_config_frame.pack(fill=tk.X, pady=(0, 8))
            self.custom_config_visible = True
        self._sync_horizon_ui()

    # --------------------------------------------------------------- submit
    def _on_submit(self):
        if self._running:
            return
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Enter a ticker.")
            return
        try:
            horizon_name = self.horizon_var.get()
            if horizon_name != "Custom":
                self._apply_horizon_preset(horizon_name)
                self._sync_horizon_ui()
            epochs   = self.epochs_var.get()
            lr       = self.lr_var.get()
            split    = self.split_var.get()
            lookback = self.lookback_var.get()
            assert epochs > 0 and lr > 0 and 0 < split < 1 and lookback >= 2
        except Exception:
            messagebox.showerror("Error",
                "Invalid parameters. Epochs > 0, LR > 0, "
                "Split in (0,1), Lookback ≥ 2.")
            return

        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="normal")
        self.log_text.configure(height=18)
        self._last_report_height_key = None
        if hasattr(self, "console_listbox"):
            self.console_listbox.delete(0, tk.END)
        self.progress_var.set(0)
        # Clear training/evaluation figures
        if hasattr(self, "fig_train"):
            self.fig_train.clear()
            self.canvas_train.draw_idle()
        if hasattr(self, "fig_eval"):
            self.fig_eval.clear()
            self.canvas_eval.draw_idle()
        self._eval_chart_payload = None
        self._eval_sr_cache.clear()
        self._set_eval_tf_buttons_enabled(False)
        if hasattr(self, "eval_accuracy_label"):
            self.eval_accuracy_label.config(text="Directional Accuracy: -")
        if hasattr(self, "eval_prev_price_label"):
            self.eval_prev_price_label.config(text="Prev Close: -")
        if hasattr(self, "eval_prev_move_label"):
            self.eval_prev_move_label.config(text="Change: -")
        if hasattr(self, "eval_current_price_label"):
            self.eval_current_price_label.config(text="$-")
        if hasattr(self, "eval_current_meta_label"):
            self.eval_current_meta_label.config(
                text=self.ticker_var.get().strip().upper()
            )
        if hasattr(self, "eval_pred_move_label"):
            self.eval_pred_move_label.config(text="Next-Day Move: -")
        if hasattr(self, "last_session_card") and self.last_session_card.winfo_ismapped():
            self.last_session_card.pack_forget()
        if hasattr(self, "training_live_row") and not self.training_live_row.winfo_ismapped():
            try:
                if hasattr(self, "last_session_card"):
                    self.training_live_row.pack(
                        fill=tk.BOTH, expand=True, pady=(0, 10), before=self.last_session_card
                    )
                else:
                    self.training_live_row.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            except Exception:
                self.training_live_row.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        if hasattr(self, "status_card") and not self.status_card.winfo_ismapped():
            self.status_card.pack(fill=tk.X, pady=(8, 0))
        if hasattr(self, "progress") and not self.progress.winfo_ismapped():
            self.progress.pack(fill=tk.X)
        self._set_eval_loading(True)
        self._set_status_pct(0)
        self._running = True
        self.submit_btn.state(["disabled"])

        period_label = self.period_var.get()
        period_code  = self.period_options.get(period_label, "5y")
        loss_fn_name = self.loss_var.get()

        threading.Thread(
            target=self._run,
            args=(ticker, epochs, lr, split, lookback,
                  period_code, period_label, loss_fn_name, horizon_name),
            daemon=True).start()

    # ============================================================= pipeline
    def _run(self, ticker, epochs, lr, train_ratio, lookback,
             period_code, period_label, loss_fn_name, horizon_name):
        try:
            # fetch data for the ticker
            self.progress_var.set(5)
            self._set_status_pct(5)

            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period_code)
            if hist.empty:
                raise ValueError(f"No data found for '{ticker}'.")

            self.progress_var.set(10)
            self._set_status_pct(10)

            close  = hist["Close"]
            volume = hist["Volume"]
            current_price = close.iloc[-1]
            prev_price    = close.iloc[-2] if len(close) > 1 else np.nan
            current_date  = close.index[-1].strftime("%Y-%m-%d")

            # ── 2  Technical indicators ────────────────────────────────

            # Use forgiving min_periods so shorter timeframes still retain usable rows.
            ma_5   = close.rolling(5, min_periods=5).mean()
            ma_20  = close.rolling(20, min_periods=10).mean()
            ma_50  = close.rolling(50, min_periods=20).mean()
            ma_200 = close.rolling(200).mean()

            rsi_series = calculate_rsi(close, 14)
            macd_line, macd_signal, macd_hist_s = calculate_macd(close)

            self.progress_var.set(15)
            self._set_status_pct(15)

            # ── 3  Build feature matrix ────────────────────────────────

            df = pd.DataFrame(index=close.index)
            df["Close"]        = close
            df["RSI"]          = rsi_series
            df["MACD_Hist"]    = macd_hist_s
            df["MA5_Ratio"]    = close / ma_5
            df["MA20_Ratio"]   = close / ma_20
            df["MA50_Ratio"]   = close / ma_50
            df["Volatility"]   = close.rolling(10).std()
            df["Pct_Change"]   = close.pct_change()
            df["Ret_5d"]       = close.pct_change(5)
            df["Ret_20d"]      = close.pct_change(20)
            df["Volume_Ratio"] = volume / volume.rolling(20).mean()
            n_3mo = min(63, len(close))
            range_min_periods = min(20, n_3mo) if n_3mo > 0 else 1
            roll_hi = close.rolling(n_3mo, min_periods=range_min_periods).max()
            roll_lo = close.rolling(n_3mo, min_periods=range_min_periods).min()
            range_denom = (roll_hi - roll_lo).replace(0, np.nan)
            df["Range_Pos_3mo"] = ((close - roll_lo) / range_denom * 100).clip(0, 100)
            df["Target_Return"] = close.pct_change().shift(-1)

            feature_cols = [
                "RSI", "MACD_Hist", "MA5_Ratio", "MA20_Ratio",
                "MA50_Ratio", "Volatility", "Pct_Change",
                "Ret_5d", "Ret_20d",
                "Volume_Ratio", "Range_Pos_3mo",
            ]
            df.dropna(inplace=True)

            X_flat = df[feature_cols].values
            y_flat = df["Target_Return"].values

            self.progress_var.set(18)
            self._set_status_pct(18)

            # ── 4  Scale + sequence build (robust for short periods) ────

            n = len(X_flat)
            val_ratio  = (1 - train_ratio) / 2
            test_ratio = (1 - train_ratio) / 2

            if n == 0:
                raise ValueError(
                    "No usable samples after feature engineering. "
                    "Try a longer timeframe (e.g., 6mo/1y) or reduce lookback."
                )

            if n <= lookback:
                raise ValueError(
                    f"Not enough rows after feature engineering for lookback={lookback}. "
                    f"Usable rows: {n}. Try a longer timeframe or smaller lookback."
                )

            # Fit scaler on the chronological train portion only.
            train_end_flat = max(1, int(n * train_ratio))
            X_flat_train = X_flat[:train_end_flat]

            scaler = StandardScaler()
            X_flat_train_s = scaler.fit_transform(X_flat_train)
            _ = X_flat_train_s  # keep variable assignment explicit for clarity
            X_flat_s = scaler.transform(X_flat)

            # Build full sequence set first, then split sequences chronologically.
            X_seq_all, y_seq_all = create_sequences(X_flat_s, y_flat, lookback)
            m = len(X_seq_all)
            if m < 3:
                raise ValueError(
                    f"Not enough sequences for Train/Val/Test split with lookback={lookback}. "
                    f"Total sequences: {m}. Try a longer timeframe or smaller lookback."
                )

            train_end_seq = max(1, int(m * train_ratio))
            val_count_seq = max(1, int(m * val_ratio))
            val_end_seq = train_end_seq + val_count_seq
            if val_end_seq >= m:
                val_end_seq = m - 1

            X_train_seq, y_train = X_seq_all[:train_end_seq], y_seq_all[:train_end_seq]
            X_val_seq, y_val = X_seq_all[train_end_seq:val_end_seq], y_seq_all[train_end_seq:val_end_seq]
            X_test_seq, y_test = X_seq_all[val_end_seq:], y_seq_all[val_end_seq:]

            if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
                raise ValueError(
                    f"Insufficient split sizes after sequence build (lookback={lookback}). "
                    f"Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}. "
                    f"Try a longer timeframe or smaller lookback."
                )

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train_seq)
            y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
            X_val_t   = torch.FloatTensor(X_val_seq)
            y_val_t   = torch.FloatTensor(y_val).reshape(-1, 1)
            X_test_t  = torch.FloatTensor(X_test_seq)
            y_test_t  = torch.FloatTensor(y_test).reshape(-1, 1)

            # DataLoader (shuffle training sequences)
            batch_size = 64
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader  = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)

            self.progress_var.set(25)
            self._set_status_pct(25)

            # ── 5  LSTM model, loss, optimizer, scheduler ──────────────
            input_dim = len(feature_cols)
            hidden_size = 64
            num_layers  = 2

            model = LSTMModel(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lstm_dropout=0.2,      # between stacked LSTM layers
                output_dropout=0.3,    # after final hidden state
            )

            if loss_fn_name == "Huber Loss":
                criterion = nn.HuberLoss(delta=1.0)
            else:
                criterion = nn.MSELoss()

            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-3)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5)
            lr_history = [lr]

            # ── 6  Training loop ───────────────────────────────────────
            self._set_status_pct(25)
            train_losses = []
            val_losses   = []
            val_epochs   = []
            console_lines = []
            actual_epochs = epochs

            early_stop = EarlyStopping(patience=15)

            # Live training chart (shown in Training tab only during active run).
            self.fig_train.clear()
            ax_live = self.fig_train.add_subplot(1, 1, 1)
            ax_live.set_title("Training Progress - LSTM Loss Curves", fontsize=9)
            ax_live.set_xlabel("Epoch", fontsize=8)
            ax_live.set_ylabel("Loss", fontsize=8)
            ax_live.grid(True, alpha=0.2)
            line_train, = ax_live.plot(
                [], [], lw=1.0, color="#2196f3", label="Training Loss"
            )
            line_val, = ax_live.plot(
                [], [], lw=1.0, color="#ff9800",
                marker="o", markersize=3, label="Validation Loss"
            )
            ax_live.legend(fontsize=7, loc="upper right")
            self.canvas_train.draw_idle()

            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss_sum = 0.0
                epoch_samples  = 0
                for X_batch, y_batch in train_loader:
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss_sum += loss.item() * len(X_batch)
                    epoch_samples  += len(X_batch)

                avg_train_loss = epoch_loss_sum / epoch_samples
                train_losses.append(avg_train_loss)

                compute_val = (epoch == 1 or epoch % 10 == 0
                               or epoch == epochs)
                if compute_val:
                    model.eval()
                    with torch.no_grad():
                        v_pred = model(X_val_t)
                        v_loss = criterion(v_pred, y_val_t).item()
                    val_losses.append(v_loss)
                    val_epochs.append(epoch)

                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(v_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr != prev_lr:
                        lr_history.append(new_lr)

                    status_line = (
                        f"Epoch {epoch}/{epochs}  "
                        f"Train {avg_train_loss:.6f}  "
                        f"Val {v_loss:.6f}"
                    )
                    console_lines.append(status_line)
                    self._console_log(status_line)

                    if early_stop(v_loss, model, epoch):
                        actual_epochs = epoch
                        break

                if epoch % 10 == 0 or epoch == epochs or epoch == actual_epochs:
                    ep_range = list(range(1, epoch + 1))
                    line_train.set_data(ep_range, train_losses)
                    line_val.set_data(val_epochs, val_losses)
                    ax_live.set_xlim(1, max(epoch, 2))
                    all_losses = train_losses + val_losses
                    if all_losses:
                        lo = min(all_losses) * 0.9
                        hi = max(all_losses) * 1.1
                        if lo == hi:
                            hi = lo + 1e-6
                        ax_live.set_ylim(lo, hi)
                    ax_live.legend(fontsize=7, loc="upper right")
                    self.canvas_train.draw_idle()

                self.progress_var.set(25 + 45 * epoch / epochs)
                self._set_status_pct(25 + 45 * epoch / epochs)

            if early_stop.stopped:
                console_lines.append(
                    f"Auto-stopped at epoch {early_stop.stop_epoch} "
                    f"after {early_stop.patience} patience checks."
                )
            console_snapshot = "\n".join(console_lines)

            early_stop.restore_best(model)

            if early_stop.best_epoch > 0:
                ax_live.axvline(
                    early_stop.best_epoch, color="#00c853",
                    ls="--", lw=0.8, alpha=0.7,
                    label=f"Best (epoch {early_stop.best_epoch})"
                )
            if early_stop.stopped:
                ax_live.axvline(
                    early_stop.stop_epoch, color="#ef5350",
                    ls="--", lw=0.8, alpha=0.7,
                    label=f"Auto-Stop (epoch {early_stop.stop_epoch})"
                )
            ax_live.legend(fontsize=7, loc="upper right")
            self.canvas_train.draw_idle()

            self.progress_var.set(70)
            self._set_status_pct(70)

            # ── 7  Evaluate on test set ────────────────────────────────
            model.eval()
            with torch.no_grad():
                test_pred_t  = model(X_test_t)
                test_loss    = criterion(test_pred_t, y_test_t).item()
                train_pred_t = model(X_train_t)
                final_train_loss = criterion(
                    train_pred_t, y_train_t).item()
                val_pred_t   = model(X_val_t)
                final_val_loss = criterion(val_pred_t, y_val_t).item()

            test_pred = test_pred_t.numpy().flatten()

            test_mae  = np.mean(np.abs(y_test - test_pred))
            test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
            ss_res = np.sum((y_test - test_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            test_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            actual_dir = np.sign(y_test)
            pred_dir   = np.sign(test_pred)
            dir_accuracy = np.mean(actual_dir == pred_dir) * 100

            tp = int(np.sum((actual_dir > 0) & (pred_dir > 0)))
            tn = int(np.sum((actual_dir <= 0) & (pred_dir <= 0)))
            fp = int(np.sum((actual_dir <= 0) & (pred_dir > 0)))
            fn = int(np.sum((actual_dir > 0) & (pred_dir <= 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score  = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            gen_gap = test_loss - final_train_loss

            # Train & Val directional accuracy (for overfitting dashboard)
            train_pred_np = train_pred_t.numpy().flatten()
            val_pred_np   = val_pred_t.numpy().flatten()
            train_dir_acc = (np.mean(
                np.sign(y_train) == np.sign(train_pred_np)) * 100)
            val_dir_acc   = (np.mean(
                np.sign(y_val) == np.sign(val_pred_np)) * 100)

            # Current prediction (last lookback window)
            latest_window = scaler.transform(
                df[feature_cols].iloc[-lookback:].values)
            latest_t = torch.FloatTensor(latest_window).unsqueeze(0)
            with torch.no_grad():
                predicted_return = model(latest_t).item()

            self.progress_var.set(80)
            self._set_status_pct(80)

            # ── 8  Feature importance via input gradient attribution ───
            # Feed training data through and compute gradient of output
            # w.r.t. each input feature, averaged across samples.
            model.eval()
            sample_t = X_train_t[:min(200, len(X_train_t))].clone()
            sample_t.requires_grad_(True)
            out = model(sample_t)
            out.sum().backward()
            grad = sample_t.grad.abs().mean(dim=(0, 1)).numpy()
            # grad shape: (n_features,)
            signed_grad = sample_t.grad.mean(dim=(0, 1)).numpy()

            weight_pairs = list(
                zip(feature_cols, grad, signed_grad))
            weight_pairs.sort(key=lambda x: x[1], reverse=True)
            fi_max_imp = max((wp[1] for wp in weight_pairs), default=0.0)
            fi_tiny_eps = 1e-12
            fi_top_pairs = weight_pairs[:8]
            fi_top_up = max(
                (wp for wp in fi_top_pairs if wp[2] > 0),
                key=lambda x: x[1],
                default=None,
            )
            fi_top_down = max(
                (wp for wp in fi_top_pairs if wp[2] < 0),
                key=lambda x: x[1],
                default=None,
            )
            fi_strongest = fi_top_pairs[0] if fi_top_pairs else None

            # ── 9  Header metrics (Evaluation cards) ───────────────────
            if hasattr(self, "eval_accuracy_label"):
                self.eval_accuracy_label.config(
                    text=f"Directional Accuracy: {dir_accuracy:.1f}%"
                )

            # Build evaluation report content.
            self._set_status_pct(80)
            self.log_text.delete("1.0", tk.END)

            training_config_lines = [
                "Model Type:       LSTM (Long Short-Term Memory)",
                f"Ticker:           {ticker}",
                f"Time Horizon:     {horizon_name}",
                f"Data Period:      {period_label} ({len(hist)} trading days)",
                f"Date:             {current_date}",
                "Architecture:     LSTM("
                f"input={input_dim}, hidden={hidden_size}, "
                f"layers={num_layers}, inter-layer dropout=0.2)"
                f" -> Dropout(0.3) -> Linear({hidden_size},1)",
                f"Lookback Window:  {lookback} days",
                "Loss Function:    "
                f"{loss_fn_name}"
                f"{'  (delta=1.0)' if loss_fn_name == 'Huber Loss' else ''}",
                "Optimizer:        Adam (weight_decay=1e-3, L2 regularization)",
                f"Initial LR:       {lr}",
                "LR Scheduler:     ReduceLROnPlateau (factor=0.5, patience=5)",
                f"Batch Size:       {batch_size}",
                "Epochs Run:       "
                f"{actual_epochs}{' (auto-stopped)' if early_stop.stopped else ''}"
                f" / {epochs} max",
                f"Early Stopping:   patience={early_stop.patience}",
                f"Best Val Epoch:   {early_stop.best_epoch} (val loss {early_stop.best_loss:.6f})"
                " - weights restored",
                f"Split:            Train {train_ratio*100:.0f}% / Val {val_ratio*100:.0f}% / Test {test_ratio*100:.0f}%",
                f"Seq. Samples:     Train {len(X_train_seq)} | Val {len(X_val_seq)} | Test {len(X_test_seq)}",
                "Feature Scaling:  StandardScaler (scikit-learn)",
                "Data Shuffling:   DataLoader(shuffle=True)",
            ]
            if len(lr_history) > 1:
                lr_steps = " -> ".join(f"{x:.1e}" for x in lr_history)
                training_config_lines.insert(
                    11, f"LR Steps:         {lr_steps}")

            # Training/validation loss history now lives in the
            # Training Configuration detail popup (plain text).
            training_config_lines.extend([
                "",
                "Training / Validation Loss Log",
                "--------------------------------",
                f"{'Epoch':>6}   {'Train Loss':>12}   {'Val Loss':>12}   {'Patience':>10}   {'Status'}",
            ])
            running_patience = 0
            running_best = None
            for i, ep in enumerate(val_epochs):
                t_loss = train_losses[ep - 1]
                v_loss = val_losses[i]
                if running_best is None or v_loss < running_best:
                    running_best = v_loss
                    running_patience = 0
                else:
                    running_patience += 1
                ratio = v_loss / t_loss if t_loss > 0 else 1.0
                if ratio > 1.5:
                    status = "Overfitting risk"
                elif ratio < 0.8:
                    status = "Underfitting risk"
                else:
                    status = "Healthy"
                best_marker = "  *" if ep == early_stop.best_epoch else ""
                training_config_lines.append(
                    f"{ep:>6}   {t_loss:>12.6f}   {v_loss:>12.6f}   "
                    f"{running_patience:>4}/{early_stop.patience:<4}  {status}{best_marker}"
                )
            training_config_lines.append(
                f"\n* = Best validation loss (epoch {early_stop.best_epoch}, "
                f"val loss {early_stop.best_loss:.6f})"
            )
            if early_stop.stopped:
                training_config_lines.append(
                    f"Auto-stopped at epoch {early_stop.stop_epoch} - val loss stalled "
                    f"for {early_stop.patience} checks."
                )
            training_config_text = "\n".join(training_config_lines)

            self._training_config_text = training_config_text

            # -- Scorecard --
            scorecard_lines = [
                "Model Performance Scorecard (Test Set)",
                "",
                f"Mean Absolute Error (MAE):      {test_mae:.6f}",
                f"Root Mean Squared Error (RMSE): {test_rmse:.6f}",
                f"Test Loss ({loss_fn_name}):     {test_loss:.6f}",
                f"R-Squared (R²):                 {test_r2:.4f}",
                "",
                "Final Losses:",
                f"  Train:      {final_train_loss:.6f}",
                f"  Validation: {final_val_loss:.6f}",
                f"  Test:       {test_loss:.6f}",
                "",
                f"Directional Accuracy:           {dir_accuracy:.1f}%",
                f"Precision (Up predictions):    {precision:.3f}",
                f"Recall (Up capture):           {recall:.3f}",
                f"F1 Score:                      {f1_score:.3f}",
                "",
                "Trading Call Quality Map (UP/DOWN)",
                "---------------------------------",
                "Predicted UP   Predicted DOWN",
                f"Actual UP         {tp:>8}          {fn:>8}",
                f"Actual DOWN       {fp:>8}          {tn:>8}",
                "",
                f"True Positives  (Up->Up):    {tp}",
                f"True Negatives  (Down->Down): {tn}",
                f"False Positives (Down->Up):   {fp}",
                f"False Negatives (Up->Down):   {fn}",
            ]
            top_raw_pairs = weight_pairs[:5]
            if top_raw_pairs:
                scorecard_lines.extend([
                    "",
                    "Top Feature Raw Gradient Values (Reference)",
                    "-------------------------------------------",
                ])
                for rank, (fname, imp, signed) in enumerate(top_raw_pairs, 1):
                    display = FEATURE_DISPLAY.get(fname, fname)
                    scorecard_lines.append(
                        f"{rank:>2}. {display:<30} abs={imp:>10.2e}  signed={signed:>+10.2e}"
                    )
            scorecard_text = "\n".join(scorecard_lines)
            self._model_scorecard_text = scorecard_text

            # Generalization gap + overfitting dashboard consolidated into a popup
            gap_msg = ""
            if gen_gap > 0 and gen_gap > final_train_loss * 0.5:
                gap_msg = "OVERFITTING DETECTED"
            elif gen_gap > 0:
                gap_msg = "Mild generalization gap"
            else:
                gap_msg = "No overfitting detected"

            tv_dir_gap = abs(train_dir_acc - val_dir_acc)

            variance_lines = []
            if tv_dir_gap > 10:
                variance_lines.append(
                    f"HIGH VARIANCE DETECTED - Train vs Val directional accuracy gap is {tv_dir_gap:.1f}% (> 10%)"
                )
                variance_lines.append("The model memorizes training patterns that don't generalize.")
                variance_lines.append(
                    "Try: reduce lookback window, increase dropout, add more data."
                )
            elif tv_dir_gap > 5:
                variance_lines.append(
                    f"Moderate variance - Train vs Val gap is {tv_dir_gap:.1f}%"
                )
            else:
                variance_lines.append(
                    f"Low variance - Train vs Val gap is {tv_dir_gap:.1f}% (healthy)"
                )

            generalization_details_lines = [
                "Generalization Gap & Overfitting Details",
                "",
                f"Train Loss (final):   {final_train_loss:.6f}",
                f"Test Loss:            {test_loss:.6f}",
                f"Gap (Test - Train):   {gen_gap:+.6f}",
                f"Status:               {gap_msg}",
                "",
                "Overfitting Dashboard",
                "Metric Summary (Train / Val / Test)",
                f"Loss:               {final_train_loss:<10.6f}  {final_val_loss:<10.6f}  {test_loss:<10.6f}",
                f"Directional Accuracy: {train_dir_acc:<10.1f}  {val_dir_acc:<10.1f}  {dir_accuracy:<10.1f}",
                "",
            ]
            generalization_details_lines.extend(variance_lines)
            generalization_details_text = "\n".join(generalization_details_lines)

            self._generalization_details_text = generalization_details_text

            # -- Market Snapshot & Generic Outlook --
            sr_levels = max(1, min(3, int(self.eval_sr_levels_var.get())))

            def _nearest_levels_for_days(days):
                window_days = min(days, len(close))
                if window_days <= 0:
                    return None, None
                supports, resistances = find_support_resistance(
                    close.iloc[-window_days:], window=20, num_levels=sr_levels
                )
                supports = sorted(supports)
                resistances = sorted(resistances)
                below_supports = [s for s in supports if s <= current_price]
                above_resistances = [r for r in resistances if r >= current_price]
                nearest_s = (
                    max(below_supports) if below_supports else (supports[-1] if supports else None)
                )
                nearest_r = (
                    min(above_resistances)
                    if above_resistances
                    else (resistances[0] if resistances else None)
                )
                return nearest_s, nearest_r

            sr_3m_support, sr_3m_resistance = _nearest_levels_for_days(63)
            sr_6m_support, sr_6m_resistance = _nearest_levels_for_days(126)
            sr_1y_support, sr_1y_resistance = _nearest_levels_for_days(252)

            # Use 6-month levels for risk/proximity calculations, with fallbacks.
            nearest_support = (
                sr_6m_support
                if sr_6m_support is not None
                else (sr_3m_support if sr_3m_support is not None else sr_1y_support)
            )
            nearest_resistance = (
                sr_6m_resistance
                if sr_6m_resistance is not None
                else (sr_3m_resistance if sr_3m_resistance is not None else sr_1y_resistance)
            )

            pred_pct = float(predicted_return * 100.0)
            dist_to_support_pct = (
                ((current_price - nearest_support) / current_price) * 100.0
                if nearest_support is not None and current_price != 0
                else None
            )
            dist_to_resistance_pct = (
                ((nearest_resistance - current_price) / current_price) * 100.0
                if nearest_resistance is not None and current_price != 0
                else None
            )

            confidence_label = (
                "High"
                if dir_accuracy >= 58
                else ("Moderate" if dir_accuracy >= 52 else "Low")
            )

            risk_score = 0
            if dir_accuracy < 52:
                risk_score += 1
            if abs(pred_pct) >= 2.0:
                risk_score += 1
            if dist_to_resistance_pct is not None and pred_pct > 0 and dist_to_resistance_pct <= 1.0:
                risk_score += 1
            if dist_to_support_pct is not None and pred_pct < 0 and dist_to_support_pct <= 1.0:
                risk_score += 1
            if dir_accuracy >= 58 and 0.2 <= abs(pred_pct) <= 1.5:
                risk_score -= 1

            if risk_score >= 2:
                risk_label = "Risky"
                risk_tag = "red"
            elif risk_score <= 0:
                risk_label = "Safer"
                risk_tag = "green"
            else:
                risk_label = "Neutral"
                risk_tag = "yellow"

            self._section("MARKET SNAPSHOT & GENERIC OUTLOOK")
            self._log(
                "  - MARKET SNAPSHOT",
                "summary",
            )
            self._log(
                f"    * Current price: ${current_price:.2f}"
            )
            self._log(
                f"    * Directional accuracy: {dir_accuracy:.1f}%"
            )
            self._log(
                f"    * Predicted next-day move: {pred_pct:+.2f}%"
            )
            weak_signal_suffix = " (weak signal)" if fi_max_imp <= fi_tiny_eps else ""
            if fi_top_up is not None:
                up_name = FEATURE_DISPLAY.get(fi_top_up[0], fi_top_up[0])
                up_pct = (
                    (fi_top_up[1] / fi_max_imp) * 100.0
                    if fi_max_imp > fi_tiny_eps
                    else 0.0
                )
                self._log(
                    f"    * Best pull-up feature: {up_name}{weak_signal_suffix}"
                )
            if fi_top_down is not None:
                down_name = FEATURE_DISPLAY.get(fi_top_down[0], fi_top_down[0])
                down_pct = (
                    (fi_top_down[1] / fi_max_imp) * 100.0
                    if fi_max_imp > fi_tiny_eps
                    else 0.0
                )
                self._log(
                    f"    * Best drawdown feature: {down_name}{weak_signal_suffix}"
                )
            self._log(
                "    * 3 months support/resistance: "
                + (f"{sr_3m_support:.2f}" if sr_3m_support is not None else "N/A")
                + " / "
                + (f"{sr_3m_resistance:.2f}" if sr_3m_resistance is not None else "N/A")
            )
            self._log(
                "    * 6 months support/resistance: "
                + (f"{sr_6m_support:.2f}" if sr_6m_support is not None else "N/A")
                + " / "
                + (f"{sr_6m_resistance:.2f}" if sr_6m_resistance is not None else "N/A")
            )
            self._log(
                "    * 1 year support/resistance: "
                + (f"{sr_1y_support:.2f}" if sr_1y_support is not None else "N/A")
                + " / "
                + (f"{sr_1y_resistance:.2f}" if sr_1y_resistance is not None else "N/A")
            )
            if dist_to_support_pct is not None or dist_to_resistance_pct is not None:
                self._log(
                    "    * Proximity to closest support/resistances: "
                    + (
                        f"{dist_to_support_pct:.2f}% above support"
                        if dist_to_support_pct is not None
                        else "support distance N/A"
                    )
                    + " | "
                    + (
                        f"{dist_to_resistance_pct:.2f}% below resistance"
                        if dist_to_resistance_pct is not None
                        else "resistance distance N/A"
                    )
                    + "."
                )
            self._log(
                f"  - GENERIC OUTLOOK: {risk_label} ({confidence_label} confidence)",
                "summary",
            )

            # -- Feature Importance (gradient-based, user-friendly) --
            self._section("FEATURE IMPORTANCE")
            if not weight_pairs:
                self._log("  No feature attribution data available.")
            else:
                top_pairs = fi_top_pairs
                top_up = fi_top_up
                top_down = fi_top_down
                strongest_name = FEATURE_DISPLAY.get(top_pairs[0][0], top_pairs[0][0])

                if fi_max_imp <= fi_tiny_eps:
                    self._log("  Attribution signal is near zero across all features.")
                    self._log("  Interpretation: output is not strongly sensitive to inputs in this sample.")

                self._log(f"  Most influential overall feature: {strongest_name}")
                if top_up is not None:
                    self._log(
                        f"  Top upward feature: {FEATURE_DISPLAY.get(top_up[0], top_up[0])}"
                    )
                if top_down is not None:
                    self._log(
                        f"  Top downward feature: {FEATURE_DISPLAY.get(top_down[0], top_down[0])}"
                    )
                self._log("")
                self._log(
                    f"  {'#':>2}  {'Feature':<30} {'Impact Score':<25} "
                    f"{'Influence':<26} {'Confidence':<10}"
                )
                self._log(
                    "  " + "-" * 2 + "  " + "-" * 30 + " " + "-" * 25 + " "
                    + "-" * 26 + " " + "-" * 10
                )
                self._log("")
                for rank, (fname, imp, signed) in enumerate(top_pairs, 1):
                    display = FEATURE_DISPLAY.get(fname, fname)

                    rel_pct = (
                        (imp / fi_max_imp) * 100.0
                        if fi_max_imp > 0
                        else 0.0
                    )
                    bar_len = int((rel_pct / 100.0) * 16)
                    bar_len = max(0, min(16, bar_len))
                    bar = "#" * bar_len + "." * (16 - bar_len)
                    impact_score = int(round(rel_pct))

                    if signed > 0:
                        direction = "Pushes forecast up"
                        tag = "green"
                    elif signed < 0:
                        direction = "Pushes forecast down"
                        tag = "red"
                    else:
                        direction = "No clear directional push"
                        tag = "yellow"

                    if rel_pct >= 70:
                        strength = "Strong"
                    elif rel_pct >= 35:
                        strength = "Notable"
                    elif rel_pct >= 10:
                        strength = "Modest"
                    else:
                        strength = "Minimal"

                    self._log(
                        f"  {rank:>2}  {display:<30} [{bar}] {impact_score:>3}/100   "
                        f"{direction:<26} {strength:<10}",
                        tag,
                    )
                self._log("")
                self._log("  Note: Impact Score is normalized to the strongest feature in this run.")

            # -- Current Prediction --
            self._section("CURRENT MODEL PREDICTION")
            self._log(f"  Price:       ${current_price:.2f}")
            pred_tag = "green" if predicted_return > 0 else "red"
            self._log(f"  Next-Day Return Estimate:  "
                      f"{predicted_return*100:+.4f}%", pred_tag)
            if abs(predicted_return) > 0.03:
                self._log(f"  Warning: prediction exceeds +/-3% - "
                          f"unreliable estimate", "orange")

            # Update Evaluation tab price strip
            try:
                # Previous vs current
                if not np.isnan(prev_price):
                    delta_abs = current_price - prev_price
                    delta_pct = (delta_abs / prev_price) * 100 if prev_price != 0 else 0.0
                    sign_tag = "green" if delta_abs > 0 else ("red" if delta_abs < 0 else "bold")
                    self.eval_prev_price_label.config(
                        text=f"Prev Close: ${prev_price:.2f}")
                    self.eval_prev_move_label.config(
                        text=f"Change: {delta_abs:+.2f} ({delta_pct:+.2f}%)")
                    # Color the label via foreground; fall back to default if no tag color
                    color_map = {"green": "#00c853", "red": "#ef5350"}
                    fg = color_map.get(sign_tag)
                    if fg:
                        self.eval_prev_price_label.configure(foreground=fg)
                        self.eval_prev_move_label.configure(foreground=fg)

                # Current price
                self.eval_current_price_label.config(
                    text=f"${current_price:.2f}")
                if hasattr(self, "eval_current_meta_label"):
                    self.eval_current_meta_label.config(text=ticker)

                # Predicted next-day move
                move_pct = predicted_return * 100
                move_dir = "Positive" if move_pct > 0 else ("Negative" if move_pct < 0 else "Flat")
                move_color = "#00c853" if move_pct > 0 else "#ef5350" if move_pct < 0 else "#ffca28"
                self.eval_pred_move_label.config(
                    text=f"Next-Day Move: {move_dir}",
                    foreground=move_color,
                )
            except Exception:
                # If any of the labels are missing, just skip UI update.
                pass

            pred_pct = predicted_return * 100.0
            pred_move = "Positive" if pred_pct > 0 else ("Negative" if pred_pct < 0 else "Flat")
            session_data = {
                "ticker": ticker,
                "dir_accuracy": float(dir_accuracy),
                "current_price": float(current_price),
                "pred_pct": float(pred_pct),
                "pred_move": pred_move,
                "period_label": period_label,
                "horizon_name": horizon_name,
                "config_line": (
                    f"Horizon={horizon_name}, Period={period_label}, "
                    f"Epochs={epochs}, LR={lr:.4g}, Split={train_ratio:.2f}, "
                    f"Lookback={lookback}, Loss={loss_fn_name}"
                ),
                "console_text": console_snapshot,
                "train_losses": list(train_losses),
                "val_losses": list(val_losses),
                "val_epochs": list(val_epochs),
                "best_epoch": int(early_stop.best_epoch) if early_stop.best_epoch else 0,
                "stop_epoch": int(early_stop.stop_epoch) if early_stop.stopped else 0,
            }
            try:
                self.root.after(0, lambda s=session_data: self._update_last_session_card(s))
            except Exception:
                pass

            # Make evaluation report text area tall enough to show all content.
            self._autosize_eval_report_text()

            self.progress_var.set(90)
            self._set_status_pct(90)

            # Render evaluation chart payload.
            chart_payload = {
                "ticker": ticker,
                "close": close,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "ma_200": ma_200,
            }
            try:
                self.root.after(
                    0,
                    lambda payload=chart_payload: self._apply_eval_chart_payload(payload),
                )
            except Exception:
                pass

            # Hide placeholder via the Tk main thread so layout updates reliably.
            if hasattr(self, "eval_placeholder"):
                try:
                    self.root.after(0, self.eval_placeholder.pack_forget)
                except Exception:
                    pass
            try:
                self.root.after(0, lambda: self._set_eval_loading(False))
            except Exception:
                pass

            # Make the report box read-only (buttons remain visible)
            if hasattr(self, "log_text"):
                try:
                    self.log_text.configure(state="disabled")
                except Exception:
                    pass
            # Safety: ensure detail buttons are visible even if hidden earlier.
            if hasattr(self, "eval_details_frame"):
                try:
                    self.root.after(
                        0,
                        lambda: (
                            self.eval_details_frame.pack(
                                fill=tk.X, padx=4, pady=(6, 8)
                            )
                            if not self.eval_details_frame.winfo_ismapped()
                            else None
                        ),
                    )
                except Exception:
                    pass

            self.progress_var.set(100)
            self._set_status_pct(100)
            try:
                self.root.after(
                    0,
                    lambda: (
                        self.status_card.pack_forget()
                        if hasattr(self, "status_card") and self.status_card.winfo_ismapped()
                        else None
                    ),
                )
            except Exception:
                pass
            try:
                self.root.after(
                    0,
                    lambda: (
                        self.training_live_row.pack_forget()
                        if hasattr(self, "training_live_row") and self.training_live_row.winfo_ismapped()
                        else None
                    ),
                )
            except Exception:
                pass

            # Automatically navigate to Evaluation tab when training is done
            if hasattr(self, "notebook") and hasattr(self, "tab_eval"):
                try:
                    self.root.after(0, lambda: self.notebook.select(self.tab_eval))
                except Exception:
                    pass

        except Exception as e:
            self.status_var.set("Error")
            self._log(f"ERROR: {e}")
            import traceback
            self._log(traceback.format_exc())
            # Tk calls must happen on the main thread.
            err_msg = str(e)
            try:
                self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
            except Exception:
                pass
            try:
                self.root.after(0, lambda: self._set_eval_loading(False))
            except Exception:
                pass
            self.progress_var.set(0)
        finally:
            self._running = False
            self.submit_btn.state(["!disabled"])
            try:
                self.root.after(0, lambda: self._set_eval_loading(False))
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    LSTMWorkbenchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

