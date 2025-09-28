"""
MT4 Backtester Dashboard - Interactive Interface

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

# Add our modules to path
sys.path.insert(0, str(Path(__file__).parent / "mt4_backtester" / "src" / "data"))
sys.path.insert(0, str(Path(__file__).parent / "mt4_backtester" / "src" / "core"))

# Import our modules
from tick_aware_manager import TickAwareDataManager
from costs import CostModel

# Page configuration
st.set_page_config(
    page_title="MT4 Backtester Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = TickAwareDataManager()
    st.session_state.cost_model = CostModel("forex_costs.csv")
    st.session_state.trades = []
    st.session_state.backtest_running = False
    st.session_state.initial_balance = 10000
    st.session_state.equity_curve = []
    st.session_state.backtest_results = None
    st.session_state.test_mode = 'quick'  # 'quick' or 'full'

# Title and description
st.title("🚀 MT4 Backtester Dashboard")
st.markdown("### Interactive backtesting with real-time visualization")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Management",
    "⚙️ Strategy Settings",
    "▶️ Run Backtest",
    "📈 Results",
    "💾 Export"
])

# ============================================================================
# TAB 1: DATA MANAGEMENT
# ============================================================================
with tab1:
    st.header("Data Selection")

    # Get available symbols
    available_symbols = st.session_state.data_manager.get_available_symbols()

    if available_symbols:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Symbol selection dropdown
            selected_symbol = st.selectbox(
                "Symbol",
                options=available_symbols,
                index=available_symbols.index('GBPAUD') if 'GBPAUD' in available_symbols else 0,
                key="data_symbol"
            )

        with col2:
            # Strategy timeframe selection
            timeframe = st.selectbox(
                "Strategy Timeframe",
                options=["M15", "M30", "H1", "H4", "D1"],
                index=2,  # Default H1
                key="strategy_timeframe",
                help="Timeframe for calculating indicators (MA, RSI, etc.)"
            )

        with col3:
            # Test mode selection
            test_mode = st.radio(
                "Test Mode",
                options=["Quick (Last 3 Months)", "Full Range"],
                index=0,
                key="test_mode_radio",
                horizontal=True
            )
            st.session_state.test_mode = 'quick' if 'Quick' in test_mode else 'full'

        # Store selections in session state for use in Run Backtest tab
        st.session_state.selected_symbol = selected_symbol
        st.session_state.selected_timeframe = timeframe

        # Get date range for symbol
        start_date, end_date = st.session_state.data_manager.get_date_range(selected_symbol)
        st.write("---")
        st.info(f"📅 Available data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Test load section
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("Test Load Data", type="primary"):
                try:
                    with st.spinner(f"Loading {selected_symbol} tick data and rolling up to {timeframe}..."):
                        # Determine date range based on test mode
                        if st.session_state.test_mode == 'quick':
                            test_start = max(start_date, end_date - timedelta(days=90))
                            test_end = end_date
                        else:
                            test_start = start_date
                            test_end = end_date

                        # Load with tick data
                        bars, ticks = st.session_state.data_manager.get_data_with_ticks(
                            selected_symbol,
                            test_start,
                            test_end,
                            timeframe
                        )

                        st.session_state.test_result = {
                            'success': True,
                            'bars': len(bars),
                            'ticks': len(ticks),
                            'start': str(bars.iloc[0]['datetime']) if not bars.empty else 'N/A',
                            'end': str(bars.iloc[-1]['datetime']) if not bars.empty else 'N/A',
                            'has_ticks': not ticks.empty
                        }
                except Exception as e:
                    st.session_state.test_result = {
                        'success': False,
                        'error': str(e)
                    }

        with col2:
            if 'test_result' in st.session_state:
                if st.session_state.test_result['success']:
                    st.success(f"✅ Loaded {st.session_state.test_result['bars']} {timeframe} bars")
                    if st.session_state.test_result['has_ticks']:
                        st.success(f"✅ {st.session_state.test_result['ticks']:,} ticks available for execution")
                    st.info(f"Period: {st.session_state.test_result['start']} to {st.session_state.test_result['end']}")
                else:
                    st.error(f"Error: {st.session_state.test_result['error']}")
    else:
        st.error("No data found in fxData/ or tickStoryData/ folders")

    # Cache management
    st.subheader("🗄️ Cache Management")

    cache_dir = Path('.backtester_cache')
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.parquet'))
        cache_size = sum(f.stat().st_size for f in cache_files) / 1024**2

        col1, col2, col3 = st.columns(3)
        col1.metric("Cached Files", len(cache_files))
        col2.metric("Cache Size", f"{cache_size:.2f} MB")

        if col3.button("Clear Cache", type="secondary"):
            st.session_state.data_manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()

# ============================================================================
# TAB 2: STRATEGY SETTINGS
# ============================================================================
with tab2:
    st.header("Strategy Configuration")

    # Strategy selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["FairPrice Grid", "Simple MA Cross", "Custom"]
    )

    if strategy_type == "FairPrice Grid":
        st.subheader("FairPrice Grid Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Entry Settings**")
            ma_period = st.number_input("MA Period", value=200, min_value=10, max_value=1000, key="ma_period")
            trigger_pips = st.number_input("Trigger Distance (pips)", value=100, min_value=10, max_value=500, key="trigger_pips")

        with col2:
            st.markdown("**Grid Settings**")
            grid_orders = st.number_input("Grid Orders", value=10, min_value=1, max_value=50, key="grid_orders")
            grid_range = st.number_input("Grid Range (pips)", value=50, min_value=10, max_value=200, key="grid_range")

        with col3:
            st.markdown("**Filter Settings**")
            use_filter = st.checkbox("Use Trend Filter", value=True, key="use_filter")
            if use_filter:
                filter_type = st.selectbox("Filter Type", ["SMA", "RSI", "MACD", "ADX"], key="filter_type")
                filter_period = st.number_input("Filter Period", value=800 if filter_type == "SMA" else 14, key="filter_period")

    st.subheader("Risk Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        lot_size = st.number_input("Lot Size", value=0.01, min_value=0.01, max_value=10.0, step=0.01, key="lot_size")

    with col2:
        st.session_state.initial_balance = st.number_input("Initial Balance ($)", value=st.session_state.initial_balance, min_value=100)

    with col3:
        equity_stop = st.slider("Equity Stop (%)", min_value=1, max_value=50, value=5)

# ============================================================================
# TAB 3: RUN BACKTEST
# ============================================================================
with tab3:
    st.header("Run Backtest")

    # Use selections from Data tab if available
    if 'selected_symbol' in st.session_state:
        mode_text = "Quick Test (Last 3 Months)" if st.session_state.test_mode == 'quick' else "Full Range"
        st.info(f"Using: {st.session_state.selected_symbol} - {st.session_state.selected_timeframe} Strategy Timeframe - {mode_text}")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Use pre-selected values or allow override
        available_symbols = st.session_state.data_manager.get_available_symbols()

        default_symbol = st.session_state.get('selected_symbol', 'GBPAUD')
        selected_symbol = st.selectbox(
            "Symbol",
            available_symbols if available_symbols else ["No data available"],
            index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
            key="backtest_symbol"
        )

        # Timeframe selection
        default_timeframe = st.session_state.get('selected_timeframe', 'H1')
        timeframes = ["M15", "M30", "H1", "H4", "D1"]
        timeframe = st.selectbox(
            "Strategy Timeframe",
            timeframes,
            index=timeframes.index(default_timeframe) if default_timeframe in timeframes else 2,
            key="backtest_timeframe"
        )

    with col2:
        # Get actual data range for symbol
        data_start, data_end = st.session_state.data_manager.get_date_range(selected_symbol)

        # Set dates based on test mode
        if st.session_state.test_mode == 'quick':
            default_start = max(data_start, data_end - timedelta(days=90))
            default_end = data_end
        else:
            default_start = data_start
            default_end = data_end

        # Date range inputs
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=data_start,
                max_value=data_end
            )
        with col2_2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=data_start,
                max_value=data_end
            )

    # Show cost information
    if selected_symbol and selected_symbol != "No data available":
        costs = st.session_state.cost_model.get_costs(selected_symbol)

        st.info(f"""
        **Trading Costs for {selected_symbol}:**
        - Spread: {costs.spread_pips} pips
        - Commission: ${costs.commission_per_lot} per standard lot
        - For {lot_size} lots: ${costs.calculate_commission(lot_size):.2f} commission per trade
        """)

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if selected_symbol == "No data available":
            st.error("Please upload data first!")
        else:
            with st.spinner("Running backtest..."):
                # Run the backtest
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load data with tick data
                    status_text.text("Loading tick data and rolling up to strategy timeframe...")
                    progress_bar.progress(20)

                    bars_df, tick_df = st.session_state.data_manager.get_data_with_ticks(
                        selected_symbol,
                        str(start_date),
                        str(end_date),
                        timeframe
                    )

                    # Use bars for strategy signals
                    df = bars_df

                    if not tick_df.empty:
                        status_text.text(f"Loaded {len(tick_df):,} ticks and {len(bars_df)} {timeframe} bars")
                    else:
                        status_text.text(f"Loaded {len(bars_df)} {timeframe} bars (no tick data)")

                    # Get strategy parameters from session state
                    ma_period = st.session_state.get('ma_period', 200)
                    trigger_pips = st.session_state.get('trigger_pips', 100)
                    use_filter = st.session_state.get('use_filter', True)
                    filter_type = st.session_state.get('filter_type', 'SMA')
                    filter_period = st.session_state.get('filter_period', 800)
                    lot_size = st.session_state.get('lot_size', 0.01)

                    # Calculate indicators
                    status_text.text("Calculating indicators...")
                    progress_bar.progress(40)

                    df['MA_200'] = df['close'].rolling(ma_period).mean()
                    if use_filter and filter_type == "SMA":
                        df['MA_Filter'] = df['close'].rolling(filter_period).mean()

                    # Run backtest
                    status_text.text("Running strategy...")
                    progress_bar.progress(60)

                    trades = []
                    balance = st.session_state.initial_balance
                    equity_curve = []
                    grid_active = False

                    for i in range(max(ma_period, filter_period if use_filter else 0), len(df)):
                        row = df.iloc[i]

                        # Update progress
                        if i % 100 == 0:
                            progress = 60 + int((i / len(df)) * 30)
                            progress_bar.progress(progress)

                        # Skip if no MA values
                        if pd.isna(row['MA_200']):
                            continue

                        # Calculate distance from MA
                        distance_pips = abs(row['close'] - row['MA_200']) / 0.0001

                        # Entry logic
                        if not grid_active and distance_pips >= trigger_pips:
                            # Check filter
                            filter_passed = True
                            if use_filter and filter_type == "SMA":
                                if 'MA_Filter' in row:
                                    filter_passed = row['close'] > row['MA_Filter']

                            if filter_passed:
                                if row['close'] > row['MA_200']:
                                    # Sell signal
                                    trade = {
                                        'datetime': row['datetime'],
                                        'type': 'SELL',
                                        'entry': row['close'],
                                        'lots': lot_size
                                    }
                                    trades.append(trade)
                                    grid_active = True
                                    balance -= costs.calculate_commission(lot_size)

                                elif row['close'] < row['MA_200']:
                                    # Buy signal
                                    trade = {
                                        'datetime': row['datetime'],
                                        'type': 'BUY',
                                        'entry': row['close'],
                                        'lots': lot_size
                                    }
                                    trades.append(trade)
                                    grid_active = True
                                    balance -= costs.calculate_commission(lot_size)

                        # Exit logic
                        elif grid_active and distance_pips < 10:
                            if trades and 'exit' not in trades[-1]:
                                trades[-1]['exit'] = row['close']
                                trades[-1]['exit_datetime'] = row['datetime']

                                # Calculate P&L
                                entry = trades[-1]['entry']
                                exit = row['close']

                                if trades[-1]['type'] == 'BUY':
                                    pips = (exit - entry) / 0.0001
                                    profit = (exit - entry) * lot_size * 100000
                                else:
                                    pips = (entry - exit) / 0.0001
                                    profit = (entry - exit) * lot_size * 100000

                                trades[-1]['pips'] = pips
                                trades[-1]['profit'] = profit - costs.calculate_commission(lot_size)

                                balance += profit - costs.calculate_commission(lot_size)
                                grid_active = False

                        # Record equity
                        equity_curve.append({
                            'datetime': row['datetime'],
                            'balance': balance,
                            'equity': balance
                        })

                    # Store results
                    st.session_state.trades = trades
                    st.session_state.equity_curve = equity_curve
                    st.session_state.df = df
                    st.session_state.symbol = selected_symbol
                    st.session_state.final_balance = balance

                    status_text.text("Backtest complete!")
                    progress_bar.progress(100)

                    # Show quick results
                    closed_trades = [t for t in trades if 'profit' in t]

                    if closed_trades:
                        total_profit = sum(t['profit'] for t in closed_trades)
                        win_rate = len([t for t in closed_trades if t['profit'] > 0]) / len(closed_trades) * 100

                        st.success(f"""
                        ✅ Backtest Complete!
                        - Total Trades: {len(closed_trades)}
                        - Win Rate: {win_rate:.1f}%
                        - Net Profit: ${total_profit:.2f}
                        - Final Balance: ${balance:.2f}
                        """)
                    else:
                        st.info("Backtest complete. No trades executed in this period.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 4: RESULTS
# ============================================================================
with tab4:
    st.header("Backtest Results")

    if 'trades' in st.session_state and st.session_state.trades:
        # Create visualizations
        closed_trades = [t for t in st.session_state.trades if 'profit' in t]

        if closed_trades:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)

            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if t['profit'] > 0]
            losing_trades = [t for t in closed_trades if t['profit'] < 0]

            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{len(winning_trades)/total_trades*100:.1f}%")
            col3.metric("Net Profit", f"${sum(t['profit'] for t in closed_trades):.2f}")
            col4.metric("Final Balance", f"${st.session_state.final_balance:.2f}")

            # Equity curve chart
            st.subheader("📈 Equity Curve")

            equity_df = pd.DataFrame(st.session_state.equity_curve)

            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_df['datetime'],
                y=equity_df['balance'],
                mode='lines',
                name='Balance',
                line=dict(color='blue', width=2)
            ))

            # Add trade markers
            for trade in closed_trades:
                if 'entry' in trade:
                    fig_equity.add_annotation(
                        x=trade['datetime'],
                        y=st.session_state.initial_balance,  # Use a reference point
                        text='↓' if trade['type'] == 'SELL' else '↑',
                        showarrow=False,
                        font=dict(size=20, color='red' if trade['type'] == 'SELL' else 'green')
                    )

            fig_equity.update_layout(
                title="Account Balance Over Time",
                xaxis_title="Date",
                yaxis_title="Balance ($)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig_equity, use_container_width=True)

            # Price chart with trades
            st.subheader("📊 Price Chart with Trades")

            df = st.session_state.df

            fig_price = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )

            # Price and MAs
            fig_price.add_trace(
                go.Candlestick(
                    x=df['datetime'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # Add MA
            fig_price.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['MA_200'],
                    mode='lines',
                    name=f'MA({ma_period})',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

            # Add trade markers
            for trade in closed_trades:
                if 'entry' in trade:
                    # Entry marker
                    fig_price.add_trace(
                        go.Scatter(
                            x=[trade['datetime']],
                            y=[trade['entry']],
                            mode='markers',
                            name='',
                            showlegend=False,
                            marker=dict(
                                symbol='triangle-up' if trade['type'] == 'BUY' else 'triangle-down',
                                size=12,
                                color='green' if trade['type'] == 'BUY' else 'red'
                            )
                        ),
                        row=1, col=1
                    )

                    # Exit marker
                    if 'exit_datetime' in trade:
                        fig_price.add_trace(
                            go.Scatter(
                                x=[trade['exit_datetime']],
                                y=[trade['exit']],
                                mode='markers',
                                name='',
                                showlegend=False,
                                marker=dict(
                                    symbol='x',
                                    size=10,
                                    color='black'
                                )
                            ),
                            row=1, col=1
                        )

            # Volume
            fig_price.add_trace(
                go.Bar(
                    x=df['datetime'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightgray'
                ),
                row=2, col=1
            )

            fig_price.update_layout(
                title=f"{st.session_state.symbol} - Backtest Trades",
                xaxis_title="Date",
                yaxis_title="Price",
                height=600,
                hovermode='x unified',
                showlegend=False
            )

            fig_price.update_xaxes(rangeslider_visible=False)

            st.plotly_chart(fig_price, use_container_width=True)

            # Trade list
            st.subheader("📋 Trade Details")

            trades_df = pd.DataFrame(closed_trades)
            trades_df = trades_df[['datetime', 'type', 'entry', 'exit', 'pips', 'profit']]
            trades_df['profit'] = trades_df['profit'].round(2)
            trades_df['pips'] = trades_df['pips'].round(1)

            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'datetime': st.column_config.DatetimeColumn('Entry Time'),
                    'type': 'Direction',
                    'entry': st.column_config.NumberColumn('Entry', format="%.5f"),
                    'exit': st.column_config.NumberColumn('Exit', format="%.5f"),
                    'pips': st.column_config.NumberColumn('Pips', format="%.1f"),
                    'profit': st.column_config.NumberColumn('Profit ($)', format="$%.2f"),
                }
            )
    else:
        st.info("No results yet. Run a backtest first!")

# ============================================================================
# TAB 5: EXPORT
# ============================================================================
with tab5:
    st.header("Export Results")

    if 'trades' in st.session_state and st.session_state.trades:
        closed_trades = [t for t in st.session_state.trades if 'profit' in t]

        if closed_trades:
            st.subheader("📥 Download Results")

            # Prepare data for export
            export_df = pd.DataFrame(closed_trades)

            # CSV download
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Trades as CSV",
                data=csv,
                file_name=f"backtest_trades_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # Generate MT4 .set file content
            st.subheader("⚙️ Export to MT4")

            set_content = f"""
; FairPrice Expert Advisor Settings
; Generated: {datetime.now()}
; Symbol: {st.session_state.symbol}

MA_Period={ma_period}
Initial_Trigger_Pips={trigger_pips}
NumberOfPendingOrders={grid_orders}
PendingOrderRangePips={grid_range}
Lots={lot_size}
Equity_StopOut_Percent={equity_stop}
"""

            if use_filter and filter_type == "SMA":
                set_content += f"Use_Slow_MA_Filter=true\nSlow_MA_Period={filter_period}\n"
            else:
                set_content += "Use_Slow_MA_Filter=false\n"

            st.download_button(
                label="Download MT4 .set File",
                data=set_content,
                file_name=f"FairPrice_{st.session_state.symbol}.set",
                mime="text/plain"
            )

            # Performance report
            st.subheader("📊 Performance Report")

            total_trades = len(closed_trades)
            winning = len([t for t in closed_trades if t['profit'] > 0])

            report = f"""
BACKTEST REPORT
===============
Symbol: {st.session_state.symbol}
Period: {start_date} to {end_date}
Timeframe: {timeframe}

STRATEGY PARAMETERS
-------------------
MA Period: {ma_period}
Trigger Distance: {trigger_pips} pips
Grid Orders: {grid_orders}
Grid Range: {grid_range} pips
Lot Size: {lot_size}

PERFORMANCE SUMMARY
-------------------
Total Trades: {total_trades}
Winning Trades: {winning}
Losing Trades: {total_trades - winning}
Win Rate: {winning/total_trades*100:.1f}%

Net Profit: ${sum(t['profit'] for t in closed_trades):.2f}
Average Profit: ${sum(t['profit'] for t in closed_trades)/total_trades:.2f}

Initial Balance: ${st.session_state.initial_balance:.2f}
Final Balance: ${st.session_state.final_balance:.2f}
Return: {(st.session_state.final_balance - st.session_state.initial_balance)/st.session_state.initial_balance*100:.1f}%
"""

            st.text_area("Report", report, height=400)

            st.download_button(
                label="Download Full Report",
                data=report,
                file_name=f"backtest_report_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.info("No results to export. Run a backtest first!")

# Footer
st.markdown("---")
st.markdown("*MT4 Backtester Dashboard v1.0 - Built with Streamlit*")