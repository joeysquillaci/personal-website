EDU_TOPICS = [
    (
        "How to Use This Tool",
        "Configure -> Train -> Evaluate. Use Time Horizon presets for quick setup, then review the result cards and report links.",
        (
            "How to Use This Tool\n\n"
            "1) Configure\n"
            "- Enter a ticker symbol (example: SPY).\n"
            "- Choose a Time Horizon (Short, Medium, Long, or Custom).\n"
            "- For Custom, use the settings button to edit period, epochs, learning rate, split, lookback, and loss.\n\n"
            "2) Train\n"
            "- Click 'Train and Evaluate'.\n"
            "- During training, the dashboard shows a live console and a live train/validation loss chart side by side.\n"
            "- Run Status appears only while training is active.\n\n"
            "3) Evaluate\n"
            "- After training, the app automatically navigates to Evaluation.\n"
            "- Review the price cards, model outlook, and report.\n"
            "- Use the detail buttons under the report for configuration, scorecard, and generalization diagnostics."
        ),
    ),
    (
        "Time Horizon and Configuration",
        "Presets provide strong starting points; Custom mode unlocks full manual tuning.",
        (
            "Time Horizon and Configuration\n\n"
            "Preset Modes\n"
            "- Short-term: tuned for faster-changing market context.\n"
            "- Medium-term: balanced default for stability and responsiveness.\n"
            "- Long-term: more context and slower learning for broader trends.\n\n"
            "Custom Mode\n"
            "- Select 'Custom' and click the settings button.\n"
            "- Recommended order for tuning:\n"
            "  1) Period and Lookback\n"
            "  2) Epochs and Learning Rate\n"
            "  3) Train Split and Loss Function\n\n"
            "Tip\n"
            "- Start with a preset, validate results, then make 1-2 custom changes at a time."
        ),
    ),
    (
        "Input Features and Why They Matter",
        "The model combines momentum, trend, volatility, and participation signals.",
        (
            "Input Features and Why They Matter\n\n"
            "Momentum Signals\n"
            "- RSI and MACD Histogram describe short-term buying/selling pressure.\n"
            "- Pct_Change, Ret_5d, Ret_20d help capture recent return behavior.\n\n"
            "Trend Signals\n"
            "- MA5/MA20/MA50 ratios show where price is relative to moving trend anchors.\n"
            "- Range_Pos_3mo tells whether price is near the top or bottom of its recent range.\n\n"
            "Risk and Participation\n"
            "- Volatility and Volume_Ratio help detect unstable regimes and unusual activity.\n\n"
            "Scaling Note\n"
            "- Features are standardized before training so no single raw scale dominates."
        ),
    ),
    (
        "Feature Importance",
        "Each feature has a meaning, and its reported impact helps explain why the model leaned up or down.",
        (
            "Feature Importance\n\n"
            "How to read it\n"
            "- Impact Score: relative strength of influence in the current run.\n"
            "- Influence: whether the feature pushed the forecast up or down.\n"
            "- Confidence: simple strength bucket (Minimal/Modest/Notable/Strong).\n\n"
            "Feature definitions and why each matters\n"
            "- RSI: Relative Strength Index over recent sessions.\n"
            "  Why it matters: helps detect overbought/oversold momentum conditions.\n"
            "- MACD_Hist: distance between MACD line and signal line.\n"
            "  Why it matters: captures momentum acceleration or deceleration.\n"
            "- MA5_Ratio: current price divided by 5-day moving average.\n"
            "  Why it matters: shows very short-term trend extension versus mean.\n"
            "- MA20_Ratio: current price divided by 20-day moving average.\n"
            "  Why it matters: tracks near-term trend direction and trend health.\n"
            "- MA50_Ratio: current price divided by 50-day moving average.\n"
            "  Why it matters: represents intermediate trend positioning.\n"
            "- Volatility: rolling standard deviation of recent returns.\n"
            "  Why it matters: higher volatility can reduce signal reliability and raise risk.\n"
            "- Pct_Change: latest daily percent return.\n"
            "  Why it matters: provides immediate one-day momentum context.\n"
            "- Ret_5d: cumulative return over the last 5 trading days.\n"
            "  Why it matters: captures short swing behavior not visible in one day.\n"
            "- Ret_20d: cumulative return over the last 20 trading days.\n"
            "  Why it matters: adds broader monthly momentum context.\n"
            "- Volume_Ratio: current volume vs 20-day average volume.\n"
            "  Why it matters: confirms whether price moves are supported by participation.\n"
            "- Range_Pos_3mo: where price sits in its recent 3-month high/low range.\n"
            "  Why it matters: indicates whether price is near potential support or resistance zones.\n\n"
            "Practical takeaway\n"
            "- Focus on the top 2-3 features and check whether their direction agrees with market context.\n"
            "- If top features conflict and confidence is low, treat the prediction as higher uncertainty."
        ),
    ),
    (
        "How to Interpret Outputs",
        "Read cards first, then report scorecards, then feature impact for context.",
        (
            "How to Interpret Outputs\n\n"
            "Evaluation Header Cards\n"
            "- Prev Close and Change: current session context.\n"
            "- Current Price: latest market anchor.\n"
            "- Next-Day Move and Directional Accuracy: model call + historical directional hit rate.\n\n"
            "Model Evaluation Report\n"
            "- Scorecard popup: MAE, RMSE, loss, R2, directional metrics, and confusion-style call map.\n"
            "- Generalization popup: train/val/test gap diagnostics for overfitting risk.\n"
            "- Feature impact section: relative influence and direction for each feature.\n\n"
            "Practical Reading Order\n"
            "- 1) Directional Accuracy\n"
            "- 2) Generalization status\n"
            "- 3) Feature impact alignment with market context"
        ),
    ),
    (
        "Last Training Session Panel",
        "After a run, return to Training to review the previous session snapshot and open detailed artifacts.",
        (
            "Last Training Session Panel\n\n"
            "- During active training, live console + loss chart are shown.\n"
            "- After completion and redirect to Evaluation, live training widgets are hidden.\n"
            "- When you navigate back to Training, the Last Training Session card reappears.\n"
            "- Click the card to open details: console snapshot, loss graph, and full configuration line.\n\n"
            "Why this design?\n"
            "- It keeps the dashboard focused during training while preserving full traceability after the run."
        ),
    ),
]
