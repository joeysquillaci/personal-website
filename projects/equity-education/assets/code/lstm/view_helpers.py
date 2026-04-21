import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from technical_helpers import find_support_resistance


def get_eval_support_resistance(ui, close):
    """Compute S/R with the same settings used by the evaluation chart."""
    if close is None or len(close) == 0:
        return [], [], "6mo", 0, 2

    tf_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}
    tf_key = ui.eval_chart_tf_var.get()
    plot_days = min(tf_days.get(tf_key, 126), len(close))
    sr_levels = max(1, min(3, int(ui.eval_sr_levels_var.get())))

    cache_key = (
        tf_key,
        plot_days,
        sr_levels,
        len(close),
        str(close.index[-1]) if len(close.index) else "",
    )
    cached = ui._eval_sr_cache.get(cache_key)
    if cached is None:
        cached = find_support_resistance(
            close.iloc[-plot_days:], window=20, num_levels=sr_levels
        )
        ui._eval_sr_cache[cache_key] = cached
    supports, resistances = cached
    return supports, resistances, tf_key, plot_days, sr_levels


def open_eval_chart_settings(ui):
    """Popup for chart support/resistance settings."""
    top = tk.Toplevel(ui.root)
    top.title("Chart Settings")
    top.geometry("340x220")
    top.resizable(False, False)
    top.transient(ui.root)

    body = ttk.Frame(top, padding=12)
    body.pack(fill=tk.BOTH, expand=True)

    show_supports_var = tk.BooleanVar(value=ui.eval_show_supports_var.get())
    show_resistances_var = tk.BooleanVar(value=ui.eval_show_resistances_var.get())
    levels_var = tk.StringVar(value=str(ui.eval_sr_levels_var.get()))

    ttk.Checkbutton(
        body,
        text="Show supports",
        variable=show_supports_var,
    ).pack(anchor="w", pady=(0, 10))
    ttk.Checkbutton(
        body,
        text="Show resistances",
        variable=show_resistances_var,
    ).pack(anchor="w", pady=(0, 10))

    row = ttk.Frame(body)
    row.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(row, text="Levels:").pack(side=tk.LEFT)
    levels_combo = ttk.Combobox(
        row,
        values=["1", "2", "3"],
        textvariable=levels_var,
        state="readonly",
        width=6,
    )
    levels_combo.pack(side=tk.LEFT, padx=(8, 0))

    btn_row = ttk.Frame(body)
    btn_row.pack(fill=tk.X, side=tk.BOTTOM)

    def _apply():
        try:
            ui.eval_show_supports_var.set(bool(show_supports_var.get()))
            ui.eval_show_resistances_var.set(bool(show_resistances_var.get()))
            ui.eval_sr_levels_var.set(int(levels_var.get()))
            ui._save_eval_chart_settings()
            ui._redraw_eval_chart()
        finally:
            top.destroy()

    ttk.Button(btn_row, text="Cancel", command=top.destroy).pack(
        side=tk.RIGHT, padx=(6, 0)
    )
    ttk.Button(btn_row, text="Apply", command=_apply).pack(side=tk.RIGHT)


def redraw_eval_chart(ui):
    """Render evaluation price chart for currently selected timeframe."""
    if not ui._eval_chart_payload:
        return

    payload = ui._eval_chart_payload
    ticker = payload["ticker"]
    close = payload["close"]
    ma_20 = payload["ma_20"]
    ma_50 = payload["ma_50"]
    ma_200 = payload["ma_200"]

    _, _, tf_key, plot_days, _ = get_eval_support_resistance(ui, close)
    dates = close.index[-plot_days:]

    supports, resistances = [], []
    if ui.eval_show_supports_var.get() or ui.eval_show_resistances_var.get():
        supports, resistances, _, _, _ = get_eval_support_resistance(ui, close)

    ui.fig_eval.clear()
    ax_price = ui.fig_eval.add_subplot(1, 1, 1)
    ax_price.plot(dates, close.iloc[-plot_days:], lw=2.4, label="Close", color="#2196f3")
    ax_price.plot(dates, ma_20.iloc[-plot_days:], lw=0.7, label="20-day MA", alpha=0.7)
    ax_price.plot(dates, ma_50.iloc[-plot_days:], lw=0.7, label="50-day MA", alpha=0.7)
    if not np.isnan(ma_200.iloc[-plot_days:]).all():
        ax_price.plot(dates, ma_200.iloc[-plot_days:], lw=0.7, label="200-day MA", alpha=0.7)

    if ui.eval_show_supports_var.get() or ui.eval_show_resistances_var.get():
        supports_desc = sorted(supports, reverse=True)
        resistances_asc = sorted(resistances)
        x_label = dates[-1]
        if ui.eval_show_supports_var.get():
            for i, support_level in enumerate(supports_desc, 1):
                ax_price.axhline(support_level, color="#66bb6a", ls="--", lw=0.8, alpha=0.7)
                ax_price.text(
                    x_label,
                    support_level,
                    f"S{i} {support_level:.2f}",
                    color="#2e7d32",
                    fontsize=7,
                    ha="right",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
                )
        if ui.eval_show_resistances_var.get():
            for i, resistance_level in enumerate(resistances_asc, 1):
                ax_price.axhline(resistance_level, color="#ef5350", ls="--", lw=0.8, alpha=0.7)
                ax_price.text(
                    x_label,
                    resistance_level,
                    f"R{i} {resistance_level:.2f}",
                    color="#c62828",
                    fontsize=7,
                    ha="right",
                    va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
                )

    tf_label = {"1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y"}.get(tf_key, "6M")
    ax_price.set_title(f"{ticker} - Key Levels ({tf_label})", fontsize=9)
    ax_price.legend(fontsize=7, loc="upper left")
    ax_price.grid(True, alpha=0.15)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ui.fig_eval.tight_layout()
    ui.canvas_eval.draw_idle()


def update_last_session_card(ui, session):
    """Update Training tab summary card with latest completed run."""
    ui._last_session = session
    ui._last_session_pending_show = True
    if not hasattr(ui, "last_session_title"):
        return
    ui.last_session_title.config(text=f"{session['ticker']} - {session['pred_move']} outlook")
    ui.last_session_meta.config(
        text=(
            f"Directional Accuracy: {session['dir_accuracy']:.1f}%\n"
            f"Current Price: ${session['current_price']:.2f}   "
            f"Predicted Move: {session['pred_move']} ({session['pred_pct']:+.2f}%)"
        )
    )


def open_last_session_detail(ui):
    """Open popup with detailed latest session artifacts."""
    if not ui._last_session:
        messagebox.showinfo(
            "No session yet",
            "Run training once to view the last session details.",
        )
        return

    s = ui._last_session
    top = tk.Toplevel(ui.root)
    top.title("Last Training Session Details")
    top.geometry("1120x560")
    top.transient(ui.root)

    shell = ttk.Frame(top, padding=10)
    shell.pack(fill=tk.BOTH, expand=True)

    summary = ttk.Label(
        shell,
        text=(
            f"{s['ticker']}  |  Directional Accuracy {s['dir_accuracy']:.1f}%  |  "
            f"Price ${s['current_price']:.2f}  |  Predicted {s['pred_move']} ({s['pred_pct']:+.2f}%)"
        ),
        font=("Helvetica", 11, "bold"),
        justify="center",
        anchor="center",
    )
    summary.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(
        shell,
        text=f"Configuration: {s.get('config_line', '-')}",
        justify="center",
        anchor="center",
    ).pack(fill=tk.X, pady=(0, 8))

    content = ttk.Frame(shell)
    content.pack(fill=tk.BOTH, expand=True)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=1)
    content.rowconfigure(0, weight=1)

    console_card = ttk.LabelFrame(content, text="Training Console Snapshot", padding=6)
    console_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    console_text = tk.Text(console_card, wrap=tk.WORD, font=("Courier", 10))
    console_sb = ttk.Scrollbar(console_card, orient=tk.VERTICAL, command=console_text.yview)
    console_text.configure(yscrollcommand=console_sb.set)
    console_sb.pack(side=tk.RIGHT, fill=tk.Y)
    console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    console_text.insert("1.0", s.get("console_text", "No console snapshot available."))
    console_text.configure(state="disabled")

    graph_card = ttk.LabelFrame(content, text="Training Loss Snapshot", padding=6)
    graph_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    fig = Figure(figsize=(5.0, 3.8), dpi=100, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    train_losses = s.get("train_losses", [])
    val_losses = s.get("val_losses", [])
    val_epochs = s.get("val_epochs", [])
    if train_losses:
        ax.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            lw=1.2,
            color="#2196f3",
            label="Train",
        )
    if val_losses and val_epochs:
        ax.plot(
            val_epochs,
            val_losses,
            lw=1.2,
            color="#ff9800",
            marker="o",
            markersize=3,
            label="Val",
        )
    best_epoch = s.get("best_epoch", 0)
    stop_epoch = s.get("stop_epoch", 0)
    if best_epoch:
        ax.axvline(best_epoch, color="#00c853", ls="--", lw=0.9, alpha=0.7)
    if stop_epoch:
        ax.axvline(stop_epoch, color="#ef5350", ls="--", lw=0.9, alpha=0.7)
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    if train_losses or val_losses:
        ax.legend(loc="upper right", fontsize=8)
    canvas = FigureCanvasTkAgg(fig, master=graph_card)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw_idle()


def populate_education_cards(ui):
    """Populate Education cards after showing a loading state."""
    if ui._education_loaded or not hasattr(ui, "edu_cards"):
        ui._set_education_loading(False)
        return

    for _, (title, preview, details) in enumerate(ui._edu_topics):
        card = ttk.LabelFrame(ui.edu_cards, text="", padding=10)
        card.pack(fill=tk.X, pady=(0, 8))

        title_lbl = ttk.Label(
            card, text=title, font=("Helvetica", 13, "bold"), justify="left"
        )
        title_lbl.pack(anchor="w")

        body_lbl = ttk.Label(
            card, text=preview, justify="left", wraplength=980, font=("Helvetica", 11)
        )
        body_lbl.pack(anchor="w", pady=(6, 0))

        more_lbl = ttk.Label(
            card, text="Click for full details", foreground="#1976d2", font=("Helvetica", 11)
        )
        more_lbl.pack(anchor="w", pady=(8, 0))

        click_cb = lambda e, t=title, d=details: ui._popup_text(t, d, font_size=14)
        for widget in (card, title_lbl, body_lbl, more_lbl):
            widget.bind("<Button-1>", click_cb)
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass

    ui._education_loaded = True
    ui._set_education_loading(False)
