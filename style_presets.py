"""
グラフのスタイル設定(フォント・線の太さ・色など)をプリセットとして保存・適用する共通モジュール。
NMR / SAXS / GPC など各アプリから render_preset_ui() を呼び出して使う。

プリセットはアプリと同じフォルダの style_presets.json に保存する。
(Streamlit Cloud では再起動でファイルが消えることがあるため、JSONのダウンロード/読み込みも用意)
"""
import json
import os

import streamlit as st

PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_presets.json")
DEFAULT_PRESET_NAME = "初期設定"
LINE_COLOR_KEY = "__line_color__"


def init_default(key, value):
    """ウィジェット作成前に初期値を入れる。ウィジェット側では value を渡さずに key だけ指定する。"""
    if key not in st.session_state:
        st.session_state[key] = value


def _load_all():
    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_all(data):
    try:
        with open(PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _apply(values, color_keys):
    for k, v in values.items():
        if k == LINE_COLOR_KEY:
            for ck in color_keys:
                st.session_state[ck] = v
        else:
            st.session_state[k] = v
    st.toast("✅ プリセットを適用しました")


def _save(app_key, style_keys):
    name = st.session_state.get(f"preset_name_{app_key}", "").strip()
    if not name:
        st.toast("⚠️ プリセット名を入力してください")
        return
    if name == DEFAULT_PRESET_NAME:
        st.toast(f"⚠️「{DEFAULT_PRESET_NAME}」は予約されています。別の名前にしてください")
        return
    values = {k: st.session_state[k] for k in style_keys if k in st.session_state}
    if st.session_state.get(f"preset_inc_color_{app_key}", False):
        values[LINE_COLOR_KEY] = st.session_state.get(f"preset_color_{app_key}", "#000000")

    data = _load_all()
    data.setdefault(app_key, {})[name] = values
    if _save_all(data):
        st.session_state[f"preset_sel_{app_key}"] = name
        st.toast(f"✅ プリセット「{name}」を保存しました")
    else:
        st.toast("⚠️ プリセットファイルに書き込めませんでした")


def _delete(app_key, name):
    data = _load_all()
    if name in data.get(app_key, {}):
        del data[app_key][name]
        _save_all(data)
        st.session_state.pop(f"preset_sel_{app_key}", None)
        st.toast(f"🗑️ プリセット「{name}」を削除しました")


def _import(app_key):
    up = st.session_state.get(f"preset_upload_{app_key}")
    if up is None:
        return
    try:
        imported = json.loads(up.getvalue().decode("utf-8"))
        if not isinstance(imported, dict):
            raise ValueError("形式が正しくありません")
        data = _load_all()
        data.setdefault(app_key, {}).update({k: v for k, v in imported.items() if isinstance(v, dict)})
        _save_all(data)
        st.toast(f"✅ {len(imported)} 件のプリセットを読み込みました")
    except Exception as e:
        st.toast(f"⚠️ プリセットの読み込みに失敗しました: {e}")


def render_preset_ui(app_key, style_keys, defaults, color_keys, default_color="#000000", container=None):
    """
    app_key    : アプリごとの識別子 ("nmr", "saxs", "gpc" など)
    style_keys : プリセットに含めるウィジェットの key 一覧
    defaults   : 「初期設定」プリセットとして使う {key: 値}
    color_keys : 「線の色」を一括で設定する対象のウィジェット key 一覧
    """
    container = container or st.sidebar
    user_presets = _load_all().get(app_key, {})
    presets = {DEFAULT_PRESET_NAME: {**defaults, LINE_COLOR_KEY: default_color}, **user_presets}

    with container.expander("🎨 スタイルプリセット (フォント・線・色)"):
        st.caption("よく使う見た目の設定を保存して、ワンクリックで呼び出せます。")
        names = list(presets.keys())
        if st.session_state.get(f"preset_sel_{app_key}") not in names:
            st.session_state[f"preset_sel_{app_key}"] = names[0]
        sel = st.selectbox("プリセット", names, key=f"preset_sel_{app_key}")

        col_a, col_d = st.columns(2)
        with col_a:
            st.button("適用", use_container_width=True, key=f"btn_preset_apply_{app_key}",
                      on_click=_apply, args=(presets[sel], color_keys))
        with col_d:
            st.button("削除", use_container_width=True, key=f"btn_preset_del_{app_key}",
                      on_click=_delete, args=(app_key, sel), disabled=(sel == DEFAULT_PRESET_NAME))

        st.divider()
        st.text_input("新しいプリセット名", key=f"preset_name_{app_key}", placeholder="例: 論文用")
        inc_color = st.checkbox("線の色も保存する (適用時に全系列をこの色にする)", key=f"preset_inc_color_{app_key}")
        init_default(f"preset_color_{app_key}", "#000000")
        st.color_picker("保存する線の色", key=f"preset_color_{app_key}", disabled=not inc_color)
        st.button("💾 現在の設定をプリセットとして保存", use_container_width=True, key=f"btn_preset_save_{app_key}",
                  on_click=_save, args=(app_key, style_keys))

        st.divider()
        st.download_button("📥 プリセットをファイルに書き出す (.json)",
                           data=json.dumps(user_presets, ensure_ascii=False, indent=2),
                           file_name=f"style_presets_{app_key}.json", mime="application/json",
                           use_container_width=True, key=f"btn_preset_export_{app_key}",
                           disabled=not user_presets)
        st.file_uploader("📤 プリセットファイルを読み込む (.json)", type=["json"],
                         key=f"preset_upload_{app_key}", on_change=_import, args=(app_key,))
