import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import re
import pickle

# -------------------------
# Utility Functions
# -------------------------
def clip_dataframe(df, q_min, q_max):
    # 対数プロット上での線形補間による人工的なピーク（スパイク）の発生を完全に防ぐため、純粋なフィルタリングのみ実行します。
    return df[(df['q_nm'] >= q_min) & (df['q_nm'] <= q_max)]

def get_log_ticks(min_val, max_val):
    if min_val <= 0 or max_val <= 0 or min_val >= max_val:
        return None, None
    min_power = int(np.floor(np.log10(min_val)))
    max_power = int(np.ceil(np.log10(max_val)))
    tickvals = []
    ticktext = []
    for p in range(min_power, max_power + 1):
        for mult in [1, 2, 5]:
            val = mult * (10 ** p)
            if val >= min_val and val <= max_val:
                tickvals.append(val)
                if 0.001 <= val < 10000:
                    text = f"{int(val)}" if val.is_integer() or val >= 1 else f"{val:g}"
                else:
                    text = f"{mult}e{p}" if mult != 1 else f"1e{p}"
                ticktext.append(text)
    return tickvals, ticktext

def clean_filename(filename):
    name = re.sub(r'^1DConv_BC_', '', filename)
    name = re.sub(r'([-_]\d{8})?_\d{6}(\.[a-zA-Z0-9]+)?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\.[a-zA-Z0-9]+$', '', name, flags=re.IGNORECASE)
    return name

def load_data(file, q_unit="q (nm⁻¹)", wavelength=0.15418):
    try:
        content = file.getvalue().decode('utf-8', errors='replace')
        lines = content.splitlines()

        parsed_data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue

            parts = re.split(r'[,\s]+', line)
            if len(parts) >= 2:
                try:
                    q = float(parts[0])
                    i = float(parts[1])
                    parsed_data.append((q, i))
                except ValueError:
                    pass

        if not parsed_data:
            return None

        df = pd.DataFrame(parsed_data, columns=['q_raw', 'I'])

        if q_unit == "q (Å⁻¹)":
            df['q_nm'] = df['q_raw'] * 10.0
        elif q_unit == "2θ (度)":
            theta_rad = np.radians(df['q_raw'] / 2.0)
            df['q_nm'] = 4.0 * np.pi * np.sin(theta_rad) / wavelength
        else:
            df['q_nm'] = df['q_raw']

        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return None

def detect_morphology(ratios):
    eval_ratios = [r for r in ratios if r >= 0.8]
    if len(eval_ratios) < 2:
        return "Disorder", 0

    tolerance = 0.05
    models = {
        "ラメラ (Lamellar)": [1, 2, 3, 4, 5, 6, 7, 8],
        "ヘキサゴナルシリンダー (Cylinder)": [1, np.sqrt(3), np.sqrt(4), np.sqrt(7), np.sqrt(9), np.sqrt(12), np.sqrt(13)],
        "BCC球 (Spheres)": [1, np.sqrt(2), np.sqrt(3), np.sqrt(4), np.sqrt(5), np.sqrt(6), np.sqrt(7), np.sqrt(8)],
        "ジャイロイド (Gyroid)": [1, np.sqrt(4/3), np.sqrt(7/3), np.sqrt(8/3), np.sqrt(10/3), np.sqrt(11/3)]
    }

    best_match = "Disorder"
    max_score = 0
    for name, theoretical_ratios in models.items():
        score = 0
        for r in eval_ratios:
            if any(abs(r - tr) / tr <= tolerance for tr in theoretical_ratios if tr > 0):
                score += 1

        match_ratio = score / len(eval_ratios)
        if match_ratio >= 0.5 and score > max_score:
            max_score = score
            best_match = name

    return best_match, max_score

def get_closest_theoretical_ratio(ratio, tolerance=0.05):
    known_ratios = [
        (1.0, "1.000 (1)"),
        (2.0, "2.000 (2, √4)"),
        (3.0, "3.000 (3, √9)"),
        (4.0, "4.000 (4, √16)"),
        (5.0, "5.000 (5, √25)"),
        (6.0, "6.000 (6, √36)"),
        (np.sqrt(2), "1.414 (√2)"),
        (np.sqrt(3), "1.732 (√3)"),
        (np.sqrt(5), "2.236 (√5)"),
        (np.sqrt(6), "2.449 (√6)"),
        (np.sqrt(7), "2.646 (√7)"),
        (np.sqrt(8), "2.828 (√8)"),
        (np.sqrt(10), "3.162 (√10)"),
        (np.sqrt(11), "3.317 (√11)"),
        (np.sqrt(12), "3.464 (√12)"),
        (np.sqrt(4/3), "1.155 (√(4/3))"),
        (np.sqrt(7/3), "1.528 (√(7/3))"),
        (np.sqrt(8/3), "1.633 (√(8/3))"),
        (np.sqrt(10/3), "1.826 (√(10/3))"),
        (np.sqrt(11/3), "1.915 (√(11/3))"),
    ]
    best_match_str = "-"
    min_diff = float('inf')

    for val, text in known_ratios:
        diff = abs(ratio - val)
        if diff / val <= tolerance and diff < min_diff:
            min_diff = diff
            best_match_str = text

    return best_match_str

# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="SAXS Analysis App", layout="wide")
    st.title("SAXS 1次元データ解析ツール")

    st.markdown("##### データの読み込み & 復元")

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        q_unit_input = st.radio("元のX軸データの種類 (読込時に自動で q[nm⁻¹] に統一します)",
                                ["q (nm⁻¹)", "q (Å⁻¹)", "2θ (度)"], index=1, horizontal=True)
        wavelength = 0.15418
        if q_unit_input == "2θ (度)":
            wavelength = st.number_input("X線波長 λ [nm] (デフォルト: Cu-Kα = 0.15418)", value=0.15418, format="%.5f")

    with col_opt2:
        # ラジオボタンの高さに合わせるための調整
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 画面を再読み込み (データ表示が消えたり固まった場合)", use_container_width=True):
            components.html("<script>parent.window.location.reload();</script>", height=0)

    # ドラッグ＆ドロップエリアの左右の高さを完全にそろえる
    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "📁 SAXS/XRDデータ (.txt等) をドラッグ＆ドロップ",
            accept_multiple_files=True,
            type=['txt', 'asc', 'csv', 'dat']
        )
    with col_up2:
        project_file = st.file_uploader("💾 プロジェクトの復元 (.saxs) をドラッグ＆ドロップ", type=["saxs"])

    if not uploaded_files and project_file is None:
        st.markdown("""
        <style>
            div[data-testid="stFileUploader"] section {
                padding: 5rem 2rem;
                min-height: 250px;
                border-width: 2px;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            div[data-testid="stFileUploader"] section {
                padding: 1rem 1rem;
                min-height: 80px;
            }
        </style>
        """, unsafe_allow_html=True)

    data_dict = {}
    blank_filename = None

    if project_file is not None:
        try:
            project_data = pickle.loads(project_file.getvalue())
            data_dict = project_data.get('data_dict', {})
            loaded_state = project_data.get('session_state', {})

            if st.session_state.get('last_loaded_project') != project_file.name:
                for k, v in loaded_state.items():
                    st.session_state[k] = v
                st.session_state['last_loaded_project'] = project_file.name
                st.rerun()

            for name, item in data_dict.items():
                if 'blank' in item['original_name'].lower():
                    blank_filename = name
        except Exception as e:
            st.error(f"プロジェクトの読み込みに失敗しました: {e}")
    elif uploaded_files:
        for f in uploaded_files:
            original_name = f.name
            display_name = clean_filename(original_name)
            current_wv = wavelength if 'wavelength' in locals() else 0.15418
            df = load_data(f, q_unit_input, current_wv)
            if df is not None:
                data_dict[display_name] = {
                    'df': df,
                    'original_name': original_name
                }
                if 'blank' in original_name.lower():
                    blank_filename = display_name

        q_min_global = float('inf')
        q_max_global = float('-inf')
        blank_df = None
        if blank_filename in data_dict:
            blank_df = data_dict[blank_filename]['df']

        for name, item in data_dict.items():
            tmp = item['df'].copy().dropna(subset=['I'])
            if tmp.empty: continue
            cur_qmin = tmp['q_nm'].min()
            cur_qmax = tmp['q_nm'].max()
            if cur_qmin < q_min_global: q_min_global = cur_qmin
            if cur_qmax > q_max_global: q_max_global = cur_qmax

        if q_min_global == float('inf'): q_min_global = 0.0
        if q_max_global == float('-inf'): q_max_global = 10.0

        st.sidebar.markdown("---")
        st.sidebar.header("バックグラウンド(Blank)設定")
        if blank_filename:
            st.sidebar.success(f"Blankファイルを自動検知しました: {blank_filename}")
            subtract_blank = st.sidebar.checkbox("Blankを減算する", value=False)
        else:
            st.sidebar.info("Blankファイル (名前に 'blank' を含むファイル) は見つかりませんでした。")
            subtract_blank = False

        st.sidebar.markdown("---")
        st.sidebar.header("範囲指定 (X軸 q)")

        if "ui_q_min" not in st.session_state: st.session_state.ui_q_min = float(q_min_global)
        if "ui_q_max" not in st.session_state: st.session_state.ui_q_max = float(q_max_global)
        if "q_slider" not in st.session_state: st.session_state.q_slider = (st.session_state.ui_q_min, st.session_state.ui_q_max)

        def sync_slider():
            st.session_state.ui_q_min = st.session_state.q_slider[0]
            st.session_state.ui_q_max = st.session_state.q_slider[1]

        def sync_num():
            st.session_state.q_slider = (st.session_state.ui_q_min, st.session_state.ui_q_max)

        col_q_min, col_q_max = st.sidebar.columns(2)
        with col_q_min:
            st.number_input("最小値", step=1.0, key="ui_q_min", on_change=sync_num)
        with col_q_max:
            st.number_input("最大値", step=1.0, key="ui_q_max", on_change=sync_num)

        ui_q_min, ui_q_max = st.sidebar.slider(
            "スライダーで表示範囲を調整",
            min_value=float(q_min_global),
            max_value=float(q_max_global),
            key="q_slider",
            on_change=sync_slider,
            step=0.1,
            format="%.4f"
        )

        if ui_q_min >= ui_q_max:
            st.sidebar.warning("※ 最小値が最大値を超えています")
            ui_q_max = ui_q_min + 0.1

        st.sidebar.markdown("---")
        st.sidebar.header("グラフ共通設定")
        normalize_data = st.sidebar.checkbox("各グラフの最大強度を 1.0 に規格化", value=True, help="データ全体での最大強度を基準に揃えます。(Tab 1・2共通)")
        plot_type = st.sidebar.radio("描画プロット", ["Semi-log (Y軸対数)", "Log-log (両対数)", "Linear (線形)"])
        show_legend_ui = st.sidebar.checkbox("グラフの凡例を表示する", value=True)
        font_size = st.sidebar.slider("フォントサイズ", 10, 40, 18)
        line_width = st.sidebar.slider("線の太さ", 1.0, 5.0, 2.0)

        st.sidebar.markdown("---")
        st.sidebar.header("軸の目盛り間隔 (0で自動)")
        col_tx, col_ty = st.sidebar.columns(2)
        with col_tx:
            dtick_x = st.number_input("X軸", value=0.0, step=1.0, format="%.2f")
        with col_ty:
            dtick_y = st.number_input("Y軸 (線形のみ)", value=0.0, step=0.1, format="%.2f")

        st.sidebar.markdown("---")
        st.sidebar.header("画像出力・保存設定")
        img_format = st.sidebar.selectbox("ダウンロード画像の形式", ["svg", "png", "jpeg", "pdf"], index=0)
        export_width = st.sidebar.number_input("出力画像の幅 (px)", value=1200, step=100)
        export_height = st.sidebar.number_input("出力画像の高さ (px)", value=800, step=100)

        is_log_y = "log" in plot_type.lower()
        is_log_x = "log-log" in plot_type.lower()
        title_y = "<b>log (Intensity)</b>" if is_log_y else "<b>Intensity [a.u.]</b>"
        show_tk_y = not is_log_y

        x_domain_log = [np.log10(max(1e-5, ui_q_min)), np.log10(max(1e-4, ui_q_max))] if is_log_x else [ui_q_min, ui_q_max]

        # --- 第二のループ：Blank減算、全体範囲での規格化などを計算し保存 ---
        processed_data = {}
        for name, item in data_dict.items():
            df = item['df'].copy()
            if subtract_blank and blank_df is not None and name != blank_filename:
                blank_I_interp = np.interp(df['q_nm'], blank_df['q_nm'], blank_df['I'])
                df['I'] = df['I'] - blank_I_interp
                df.loc[df['I'] <= 0, 'I'] = np.nan

            # 全体での最大強度で規格化
            if normalize_data:
                ymax = df['I'].max()
                if pd.notna(ymax) and ymax > 0:
                    df['I'] = df['I'] / ymax

            processed_data[name] = df

        tab1, tab2 = st.tabs(["📊 オーバーレイ比較モード", "🔍 個別ピーク解析モード"])

        # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # タブ1: オーバーレイ比較
        # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        with tab1:
            st.header("波形のオーバーレイ比較")

            file_names = list(processed_data.keys())
            default_selections = [name for name in file_names if name != blank_filename]

            col_left, col_right = st.columns([1, 3])

            with col_left:
                st.subheader("波形のオフセット (ずらし表示)")
                stack_y = st.number_input("自動Y軸シフト", value=0.0, step=2.0, help="波形を上下にずらします。対数の場合は 10^値 の乗算になります。")

                st.markdown("---")
                st.subheader("全体ピーク検出設定")
                use_log_pk_ov = st.checkbox("📈 強度を対数変換して検出 (微小ピーク向け)", value=is_log_y, key="log_pk_ov")
                ignore_q_min_ov = st.number_input("極低角のピーク検出を除外 (q < 〇〇 は無視)", min_value=0.0, value=0.3, step=0.1, format="%.2f", key="ign_q_ov", help="ビームストップ等の不要な寄生散乱ノイズをピーク検出対象から外します。")

                # 計算用のmaxを取得
                max_all = float("-inf")
                min_all = float("inf")
                for n in file_names:
                    tmp = processed_data[n]['I'].dropna()
                    if not tmp.empty:
                        if use_log_pk_ov:
                            tmp = np.log10(tmp.clip(lower=1e-12))
                            val = tmp.max() - tmp.min()
                            max_all = max(max_all, val)
                            min_all = min(min_all, tmp.min())
                        else:
                            max_all = max(max_all, tmp.max())
                            min_all = min(min_all, tmp.min())

                if max_all == float("-inf"): max_all = 1.0; min_all = 0.0

                max_prom_ov = 1.0 if use_log_pk_ov else float(max_all) * 0.5
                prom_fmt_ov = "%.3f" if use_log_pk_ov else "%.2e"

                over_prom = st.slider("一括プロミネンス", 0.0, max_prom_ov, max_prom_ov * 0.05, format=prom_fmt_ov, key="o_prom")
                over_dist = st.slider("最小間隔(データ点数)", 1, 100, 5, key="o_dist", help="近すぎるピークを除外")
                o_marker_offset = st.slider("マーカー表示高さ(浮かせ幅)", 1.0, 30.0, 3.0, step=0.5, key="o_mof")

                st.subheader("表示対象・個別設定")
                modifiers = {}
                peak_flags = {}
                display_names = {}
                selected_files = []

                for name in file_names:
                    col_chk, col_ren = st.columns([1, 2])
                    with col_chk:
                        show_file = st.checkbox(f"表示", value=(name in default_selections), key=f"overlay_{name}")
                    with col_ren:
                        custom_n = st.text_input("系列名", value=name, key=f"ren_{name}", label_visibility="collapsed")

                    if show_file:
                        selected_files.append(name)
                        display_names[name] = custom_n
                        col_m, col_p = st.columns([1, 1])
                        with col_m:
                            modifiers[name] = st.number_input("倍率", value=1.0, step=0.5, format="%.2f", key=f"mult_{name}")
                        with col_p:
                            peak_flags[name] = st.checkbox("ピーク", value=False, key=f"pkf_{name}")

            with col_right:
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                y_min_g, y_max_g = float('inf'), float('-inf')

                for i, name in enumerate(selected_files):
                    df_full = processed_data[name].copy().dropna(subset=['I'])
                    if df_full.empty: continue

                    x_offset = 0.0
                    y_offset = stack_y * i

                    df_full['q_nm'] = df_full['q_nm'] + x_offset
                    if is_log_y:
                        df_full['I'] = df_full['I'] * (10 ** y_offset)
                    else:
                        df_full['I'] = df_full['I'] + y_offset

                    df_vis = clip_dataframe(df_full, ui_q_min, ui_q_max)

                    multiplier = modifiers.get(name, 1.0)
                    y_data_vis = df_vis['I'] * multiplier if not df_vis.empty else pd.Series(dtype=float)
                    c = colors[i % len(colors)]

                    if not y_data_vis.empty:
                        if y_data_vis.min() < y_min_g: y_min_g = y_data_vis.min()
                        if y_data_vis.max() > y_max_g: y_max_g = y_data_vis.max()

                    display_n = display_names.get(name, name)
                    fig.add_trace(go.Scatter(
                        x=df_vis['q_nm'], y=df_vis['I'] * multiplier, mode='lines', name=display_n,
                        line=dict(width=line_width, color=c)
                    ))

                    if peak_flags.get(name, False) and not y_data_vis.empty:
                        if use_log_pk_ov:
                            y_eval = np.log10(y_data_vis.clip(lower=1e-12))
                            pks, props = find_peaks(y_eval, prominence=over_prom, distance=over_dist)
                        else:
                            y_eval = y_data_vis
                            pks, props = find_peaks(y_eval, prominence=over_prom*multiplier, distance=over_dist)

                        # 指定した q値 以下の低角ノイズピークを除外
                        valid_mask = df_vis['q_nm'].iloc[pks].values >= ignore_q_min_ov
                        pks = pks[valid_mask]
                        proms = props['prominences'][valid_mask]

                        # 最も目立つピークを最大8つに制限
                        if len(pks) > 8:
                            top_idx = np.argsort(proms)[-8:]
                            pks = np.sort(pks[top_idx])

                        if len(pks) > 0:
                            p_offset = o_marker_offset if is_log_y else (y_max_g * (o_marker_offset - 1.0) * 0.1)
                            p_y = y_data_vis.iloc[pks] * p_offset if is_log_y else y_data_vis.iloc[pks] + p_offset

                            fig.add_trace(go.Scatter(
                                x=df_vis['q_nm'].iloc[pks], y=p_y,
                                mode='markers', name=f"{display_n} Peaks",
                                marker=dict(color=c, size=14, symbol='triangle-down'),
                                showlegend=False
                            ))

                t_vals_y, t_text_y = None, None
                t_vals_x, t_text_x = None, None

                if plot_type == "Semi-log (Y軸対数)":
                    fig.update_layout(yaxis_type="log")
                elif plot_type == "Log-log (両対数)":
                    fig.update_layout(xaxis_type="log", yaxis_type="log")
                    if ui_q_min > 0 and ui_q_max > ui_q_min:
                        t_vals_x, t_text_x = get_log_ticks(ui_q_min, ui_q_max)

                fig.update_layout(
                    plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=font_size, color='black'),
                    margin=dict(l=60, r=40, t=40, b=60),
                    showlegend=show_legend_ui,
                    legend=dict(x=1.05, y=1, font=dict(color='black', size=font_size))
                )

                x_dt_kwargs = {}
                if not t_vals_x and dtick_x > 0:
                    x_dt_kwargs['dtick'] = dtick_x
                    x_dt_kwargs['tickmode'] = 'linear'
                else:
                    x_dt_kwargs['tickmode'] = 'array' if t_vals_x else 'auto'

                y_dt_kwargs = {}
                if not is_log_y and dtick_y > 0:
                    y_dt_kwargs['dtick'] = dtick_y
                    y_dt_kwargs['tickmode'] = 'linear'

                fig.update_xaxes(
                    title_text="<b>q [nm<sup>-1</sup>]</b>",
                    title_font=dict(size=font_size, color='black'),
                    tickfont=dict(size=font_size, color='black'),
                    ticks='inside', tickcolor='black', tickwidth=2, ticklen=6,
                    showline=True, linecolor='black', linewidth=2,
                    showgrid=False, mirror=True, zeroline=False,
                    range=x_domain_log,
                    autorange=False,
                    tickvals=t_vals_x, ticktext=t_text_x,
                    **x_dt_kwargs
                )

                fig.update_yaxes(
                    title_text=title_y,
                    title_font=dict(size=font_size, color='black'),
                    tickfont=dict(size=font_size, color='black'),
                    showticklabels=show_tk_y,
                    ticks='inside' if show_tk_y else "", tickcolor='black', tickwidth=2, ticklen=6,
                    showline=True, linecolor='black', linewidth=2,
                    showgrid=False, mirror=True, zeroline=False,
                    **y_dt_kwargs
                )

                config = {
                    'toImageButtonOptions': {'format': img_format, 'filename': 'saxs_overlay', 'height': export_height, 'width': export_width, 'scale': 1},
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                }
                st.plotly_chart(fig, use_container_width=True, config=config)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🖼️ コピー用画像を作成 (右クリックでコピー可)", use_container_width=True, key="btn_copy_t1"):
                    with st.spinner("画像を生成中..."):
                        try:
                            img_bytes = fig.to_image(format="png", width=export_width, height=export_height, scale=3)
                            st.success("✅ 生成完了！下の画像を **右クリック →「画像をコピー」** してください。")
                            st.image(img_bytes, use_container_width=True)
                        except Exception as e:
                            if "kaleido" in str(e).lower() or "mathjax" in str(e).lower():
                                st.error("⚠️ 'kaleido' が必要です。ターミナルで `pip install -U kaleido` を実行してください。")
                            else:
                                st.error(f"エラー: {e}")

        # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # タブ2: 個別ピーク解析
        # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        with tab2:
            st.header("ピーク解析とモルフォロジー推測")

            file_names = list(processed_data.keys())

            col_controls, col_plot = st.columns([1, 2])

            with col_controls:
                controls_container = st.container()

            with col_plot:
                plot_container = st.container()

                st.markdown("---")

                if "current_file_idx" not in st.session_state:
                    st.session_state.current_file_idx = 0
                if st.session_state.current_file_idx >= len(file_names):
                    st.session_state.current_file_idx = 0

                col_prev, col_sel, col_next = st.columns([1, 4, 1])

                with col_prev:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("◀ 前へ", use_container_width=True):
                        st.session_state.current_file_idx = (st.session_state.current_file_idx - 1) % len(file_names)
                        st.rerun()

                with col_next:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("次へ ▶", use_container_width=True):
                        st.session_state.current_file_idx = (st.session_state.current_file_idx + 1) % len(file_names)
                        st.rerun()

                with col_sel:
                    chosen_name = st.selectbox(
                        "📝 解析対象のファイルを選択 (グラフの切り替え)",
                        file_names,
                        index=st.session_state.current_file_idx
                    )

                if file_names.index(chosen_name) != st.session_state.current_file_idx:
                    st.session_state.current_file_idx = file_names.index(chosen_name)
                    st.rerun()

                analysis_file = chosen_name

            if analysis_file:
                pk_key = f"manual_pks_{analysis_file}"
                if pk_key not in st.session_state: st.session_state[pk_key] = []

                df_full = processed_data[analysis_file].copy()
                df_full = df_full.dropna(subset=['I'])

                df_vis = clip_dataframe(df_full, ui_q_min, ui_q_max)
                if df_vis.empty:
                    df_vis = df_full # fallback

                with controls_container:
                    st.subheader("ピーク検出設定")
                    use_log_pk_t2 = st.checkbox("📈 強度を対数変換して検出 (高角側の小さいピークを推奨検出)", value=is_log_y, key=f"log_pk_t2_{analysis_file}")
                    ignore_q_min_t2 = st.number_input("極低角のピーク検出を除外 (q < 〇〇 は無視)", min_value=0.0, value=0.2, step=0.1, format="%.2f", key=f"ign_q_t2_{analysis_file}", help="ビームストップ等の不要な寄生散乱ノイズを自動検出対象から外します。")

                    df_tmp = df_vis['I'].dropna()
                    if use_log_pk_t2 and not df_tmp.empty:
                        y_eval_t2 = np.log10(df_tmp.clip(lower=1e-12))
                        max_I_t2 = float(y_eval_t2.max() - y_eval_t2.min())
                        min_H_t2 = float(y_eval_t2.min())
                        max_H_t2 = float(y_eval_t2.max())
                    elif not df_tmp.empty:
                        y_eval_t2 = df_tmp
                        max_I_t2 = float(df_tmp.max())
                        min_H_t2 = 0.0
                        max_H_t2 = float(max_I_t2)
                    else:
                        y_eval_t2 = pd.Series(dtype=float)
                        max_I_t2 = 1.0
                        min_H_t2 = 0.0
                        max_H_t2 = 1.0

                    max_prom_t2 = 1.0 if use_log_pk_t2 else float(max_I_t2) * 0.5
                    prom_fmt_t2 = "%.3f" if use_log_pk_t2 else "%.2e"

                    prominence = st.slider("プロミネンス", 0.0, max_prom_t2, max_prom_t2 * 0.02, format=prom_fmt_t2, key=f"t2_prom_{analysis_file}")
                    t2_dist = st.slider("最小間隔(データ点数)", 1, 100, 5, key=f"t2_dist_{analysis_file}", help="近すぎるピークを除外")
                    t2_marker_offset = st.slider("マーカー表示高さ(浮かせ幅)", 1.0, 30.0, 3.0, step=0.5, key=f"t2_mof_{analysis_file}")

                    st.markdown("---")
                    st.subheader("手動ピーク管理")
                    custom_q = st.number_input("新規追加するピークの q [nm⁻¹]", min_value=0.0, step=0.005, format="%.4f", key=f"custom_q_{analysis_file}")

                    if st.button("追加する (Add)", use_container_width=True):
                        if custom_q not in st.session_state[pk_key]:
                            st.session_state[pk_key].append(custom_q)
                            st.session_state[pk_key] = sorted(st.session_state[pk_key])
                            st.rerun()

                    if len(st.session_state[pk_key]) > 0:
                        st.markdown("**個別編集・削除**")
                        new_pks = []
                        changed = False

                        for i, mq in enumerate(st.session_state[pk_key]):
                            col_e1, col_e2 = st.columns([3, 1])
                            with col_e1:
                                edited_val = st.number_input(f"ピーク {i+1}", value=float(mq), step=0.005, format="%.4f", key=f"edit_mq_{analysis_file}_{i}", label_visibility="collapsed")
                            with col_e2:
                                if st.button("削除", key=f"del_mq_{analysis_file}_{i}", use_container_width=True):
                                    changed = True
                                    continue

                            if edited_val != mq:
                                changed = True
                            new_pks.append(edited_val)

                        if changed:
                            st.session_state[pk_key] = sorted(list(set(new_pks)))
                            st.rerun()

                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("全リセット (Clear All)", use_container_width=True):
                            st.session_state[pk_key] = []
                            st.rerun()

                    # Only find peaks in the visible data range
                    peaks, props = find_peaks(y_eval_t2.values, prominence=prominence, distance=t2_dist)

                    # 指定した q値 以下の低角ノイズピークを除外
                    valid_mask_t2 = df_vis['q_nm'].iloc[peaks].values >= ignore_q_min_t2
                    peaks = peaks[valid_mask_t2]
                    proms_t2 = props['prominences'][valid_mask_t2]

                    # 目立つピーク上位8つまでに自動制限 (大量のノイズ検出を防ぐ)
                    if len(peaks) > 8:
                        top_idx = np.argsort(proms_t2)[-8:]
                        peaks = np.sort(peaks[top_idx])

                    # Ensure alignment of indices with full df_vis if it has custom index
                    auto_qs = df_vis['q_nm'].iloc[peaks].values if len(peaks) > 0 else []

                    all_qs = np.unique(np.concatenate((auto_qs, st.session_state[pk_key])))
                    all_qs = np.sort(all_qs)

                    st.markdown("---")
                    st.subheader("個別表示のON/OFF")
                    st.caption("チェックを外すとグラフ・表から除外されます。")

                    active_qs = []

                    q1_ref = None
                    if len(all_qs) > 0:
                        with st.container(height=200):
                            for q_val in all_qs:
                                is_manual = q_val in st.session_state[pk_key]
                                label = f"q = {q_val:.4f}" + (" (手動)" if is_manual else "")
                                chk_key = f"chk_{analysis_file}_{q_val}"
                                if chk_key not in st.session_state:
                                    st.session_state[chk_key] = True

                                is_active = st.checkbox(label, value=st.session_state[chk_key], key=chk_key)
                                if is_active:
                                    active_qs.append(q_val)

                        if len(active_qs) > 0:
                            saved_q1_key = f"q1_sel_{analysis_file}"
                            
                            if saved_q1_key in st.session_state and st.session_state[saved_q1_key] in active_qs:
                                best_idx = list(active_qs).index(st.session_state[saved_q1_key])
                            else:
                                active_Is_for_q1 = np.interp(active_qs, df_full['q_nm'], df_full['I'])
                                max_idx = int(np.argmax(active_Is_for_q1))

                                # 全てのピークを順に q1 として試し、最もスコアが高いものを初期選択とする
                                best_idx = max_idx
                                best_morph_score = -1

                                for idx, cand_q1 in enumerate(active_qs):
                                    cand_ratios = active_qs / cand_q1
                                    morph, score = detect_morphology(cand_ratios)
                                    if morph != "Disorder" and score > best_morph_score:
                                        best_morph_score = score
                                        best_idx = idx

                            st.markdown("---")
                            
                            def update_q1_sel():
                                st.session_state[saved_q1_key] = st.session_state[f"ui_{saved_q1_key}"]

                            q1_ref = st.selectbox(
                                "⭐ 相対比の基準となる第1ピーク (q₁)", 
                                active_qs, 
                                index=best_idx, 
                                format_func=lambda x: f"q = {x:.4f} nm⁻¹", 
                                key=f"ui_{saved_q1_key}",
                                on_change=update_q1_sel,
                                help="ピークが追加・削除されても選択した基準ピークを維持します。最初は最も高スコアとなるピークを自動選択します。"
                            )
                            # 未保存時やピーク削除で無効化された場合は新しく保存
                            if saved_q1_key not in st.session_state or st.session_state[saved_q1_key] not in active_qs:
                                st.session_state[saved_q1_key] = q1_ref
                    else:
                        st.info("ピークが見つかりません。左パネルからピークを設定してください。")

                with plot_container:
                    active_qs = np.array(active_qs)

                    if len(active_qs) > 0:
                        # 強度補間
                        active_Is = np.interp(active_qs, df_full['q_nm'], df_full['I'])
                        q1 = q1_ref if 'q1_ref' in locals() and q1_ref is not None else active_qs[0]
                        ratios = active_qs / q1
                        d_spacings = (2.0 * np.pi) / active_qs

                        closest_strs = [get_closest_theoretical_ratio(r) for r in ratios]

                        res_df = pd.DataFrame({
                            'ピーク番号': range(1, len(active_qs) + 1),
                            'q': active_qs,
                            'd_spacing': d_spacings,
                            'Ratio': ratios,
                            'Closest': closest_strs
                        })

                        st.subheader("ピーク解析データ & モルフォロジー推定")
                        col_tbl1, col_tbl2 = st.columns([1, 1])

                        with col_tbl1:
                            fig_tbl = go.Figure(data=[go.Table(
                                columnwidth=[30, 80, 80, 80, 100],
                                header=dict(
                                    values=["<b>No.</b>", "<b>q [nm<sup>-1</sup>]</b>", "<b>d [nm]</b>", "<b>相対比</b>", "<b>近い理論比</b>"],
                                    fill_color='lightgrey', align='center',
                                    font=dict(color='black', size=14), height=30
                                ),
                                cells=dict(
                                    values=[
                                        res_df['ピーク番号'],
                                        res_df['q'].apply(lambda x: f"{x:.4f}"),
                                        res_df['d_spacing'].apply(lambda x: f"{x:.4f}"),
                                        res_df['Ratio'].apply(lambda x: f"{x:.3f}"),
                                        res_df['Closest']
                                    ],
                                    fill_color='white', align='center',
                                    font=dict(color='black', size=14), height=25
                                )
                            )])
                            fig_tbl.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=45 + len(res_df)*30, paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_tbl, use_container_width=True)

                        with col_tbl2:
                            hint_df = pd.DataFrame({
                                'LAM': ['1', '2', '3', '4', '5', '6', '7', '8'],
                                'CYL': ['1', '√3 (1.73)', '√4 (2.00)', '√7 (2.65)', '√9 (3.00)', '√12 (3.46)', '√13 (3.61)', '√16 (4.00)'],
                                'BCC': ['1', '√2 (1.41)', '√3 (1.73)', '√4 (2.00)', '√5 (2.24)', '√6 (2.45)', '√7 (2.65)', '√8 (2.83)'],
                                'GYR': ['1', '√(4/3) (1.15)', '√(7/3) (1.53)', '√(8/3) (1.63)', '√(10/3) (1.83)', '√(11/3) (1.91)', '√(12/3) (2.00)', '√(14/3) (2.16)']
                            })
                            disp_len = max(5, len(res_df))
                            hint_df_disp = hint_df.iloc[:disp_len]

                            fig_hint = go.Figure(data=[go.Table(
                                columnwidth=[60, 80, 80, 100],
                                header=dict(
                                    values=["<b>LAM</b>", "<b>CYL</b>", "<b>BCC</b>", "<b>GYR</b>"],
                                    fill_color='#e6f2ff', align='center', font=dict(color='black', size=14), height=30
                                ),
                                cells=dict(
                                    values=[hint_df_disp['LAM'], hint_df_disp['CYL'], hint_df_disp['BCC'], hint_df_disp['GYR']],
                                    fill_color='white', align='center', font=dict(color='black', size=14), height=25
                                )
                            )])
                            fig_hint.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=45 + disp_len*30, paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_hint, use_container_width=True)

                        morphology, score_val = detect_morphology(ratios)
                        st.markdown("""
                        <style>
                        div.stAlert {
                            margin-top: -10px !important;
                            padding-top: 0.5rem !important;
                            padding-bottom: 0.5rem !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        if "Disorder" in morphology:
                            st.warning(f"**💡 判定結果:** 特定のモルフォロジー構造と一致しません ({morphology})")
                        else:
                            st.success(f"**💡 判定結果:** 推定構造は **{morphology}** です！")
                    else:
                        st.info("ピークが選択されていません。左パネルからピークを設定してください。")

                    st.markdown("---")

                    fig_peak = go.Figure()
                    fig_peak.add_trace(go.Scatter(
                        x=df_vis['q_nm'], y=df_vis['I'], mode='lines', name='Data',
                        line=dict(color='black', width=line_width)
                    ))

                    if len(active_qs) > 0:
                        max_I = float(df_vis['I'].max()) if not df_vis.empty else 1.0
                        p_offset = t2_marker_offset if is_log_y else (max_I * (t2_marker_offset - 1.0) * 0.1)
                        p_y = active_Is * p_offset if is_log_y else active_Is + p_offset

                        fig_peak.add_trace(go.Scatter(
                            x=active_qs, y=p_y,
                            mode='markers', name='Selected Peaks',
                            marker=dict(color='black', size=14, symbol='triangle-down')
                        ))

                    t_vals_y, t_text_y = None, None
                    t_vals_x, t_text_x = None, None

                    if plot_type == "Semi-log (Y軸対数)":
                        fig_peak.update_layout(yaxis_type="log")
                    elif plot_type == "Log-log (両対数)":
                        fig_peak.update_layout(xaxis_type="log", yaxis_type="log")
                        if ui_q_min > 0 and ui_q_max > ui_q_min:
                            t_vals_x, t_text_x = get_log_ticks(ui_q_min, ui_q_max)

                    fig_peak.update_layout(
                        plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=font_size, color='black'),
                        margin=dict(l=60, r=40, t=40, b=60),
                        showlegend=show_legend_ui,
                        legend=dict(font=dict(color='black', size=font_size))
                    )

                    x_dt_kwargs_pk = {}
                    if not t_vals_x and dtick_x > 0:
                        x_dt_kwargs_pk['dtick'] = dtick_x
                        x_dt_kwargs_pk['tickmode'] = 'linear'
                    else:
                        x_dt_kwargs_pk['tickmode'] = 'array' if t_vals_x else 'auto'

                    y_dt_kwargs_pk = {}
                    if not is_log_y and dtick_y > 0:
                        y_dt_kwargs_pk['dtick'] = dtick_y
                        y_dt_kwargs_pk['tickmode'] = 'linear'

                    fig_peak.update_xaxes(
                        title_text="<b>q [nm<sup>-1</sup>]</b>",
                        title_font=dict(size=font_size, color='black'),
                        tickfont=dict(size=font_size, color='black'),
                        ticks='inside', tickcolor='black', tickwidth=2, ticklen=6,
                        showline=True, linecolor='black', linewidth=2,
                        showgrid=False, mirror=True, zeroline=False,
                        range=x_domain_log,
                        autorange=False,
                        tickvals=t_vals_x, ticktext=t_text_x,
                        **x_dt_kwargs_pk
                    )

                    fig_peak.update_yaxes(
                        title_text=title_y,
                        title_font=dict(size=font_size, color='black'),
                        tickfont=dict(size=font_size, color='black'),
                        showticklabels=show_tk_y,
                        ticks='inside' if show_tk_y else "", tickcolor='black', tickwidth=2, ticklen=6,
                        showline=True, linecolor='black', linewidth=2,
                        showgrid=False, mirror=True, zeroline=False,
                        **y_dt_kwargs_pk
                    )

                    config_pk = {
                        'toImageButtonOptions': {'format': img_format, 'filename': f'saxs_peak_{analysis_file}', 'height': export_height, 'width': export_width, 'scale': 1},
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                    }
                    st.plotly_chart(fig_peak, use_container_width=True, config=config_pk)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🖼️ コピー用画像を作成 (右クリックでコピー可)", use_container_width=True, key="btn_copy_t2"):
                        with st.spinner("画像を生成中..."):
                            try:
                                img_bytes = fig_peak.to_image(format="png", width=export_width, height=export_height, scale=3)
                                st.success("✅ 生成完了！下の画像を **右クリック →「画像をコピー」** してください。")
                                st.image(img_bytes, use_container_width=True)
                            except Exception as e:
                                if "kaleido" in str(e).lower() or "mathjax" in str(e).lower():
                                    st.error("⚠️ 'kaleido' が必要です。ターミナルで `pip install -U kaleido` を実行してください。")
                                else:
                                    st.error(f"エラー: {e}")

    # --- プロジェクト保存 ---
    if data_dict:
        st.sidebar.divider()
        st.sidebar.header("📦 プロジェクトの保存")

        def create_project_file():
            state_to_save = {k: v for k, v in st.session_state.items()
                             if not k.startswith("FormSubmitter")
                             and k != "last_loaded_project"
                             and not k.startswith("uploaded_")
                             and not k.startswith("up_")
                             and not k.startswith("down_")
                             and not k.startswith("btn_")
                             and not k.startswith("pkf_")
                             and not k.startswith("mult_")}

            project_data = {
                'data_dict': data_dict,
                'session_state': state_to_save
            }
            return pickle.dumps(project_data)

        if len(data_dict) == 1:
            dl_filename = f"{list(data_dict.keys())[0]}.saxs"
        else:
            names = list(data_dict.keys())
            base = "_".join(names[:3])
            if len(names) > 3:
                base += "_etc"
            dl_filename = f"{base}.saxs"

        st.sidebar.download_button(
            label="📥 現在の状態を復元用ファイル(.saxs)として保存",
            data=create_project_file(),
            file_name=dl_filename,
            mime="application/octet-stream"
        )

if __name__ == "__main__":
    main()
