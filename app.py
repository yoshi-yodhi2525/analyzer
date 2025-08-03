import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
import jieba
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import threading
from io import BytesIO
import base64

# NLTKデータのダウンロード
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTKデータを確実にダウンロード
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# 日本語フォントの設定
def setup_japanese_font():
    """日本語フォントを設定"""
    # Streamlit Cloudでは日本語フォントが利用できないため、
    # デフォルトの英語フォントのみを使用
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

# フォント設定を実行
setup_japanese_font()

# ページ設定
st.set_page_config(
    page_title="テキスト分析アプリ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # 日本語のストップワードを追加
        japanese_stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また', 'それ', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か', 'だ', 'これ', 'で', 'あ', 'や', 'られ', 'なる', 'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また', 'それ', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か', 'だ', 'これ', 'で', 'あ', 'や', 'られ', 'なる'
        }
        self.stop_words.update(japanese_stop_words)
    
    def detect_language(self, text):
        """テキストの言語を検出"""
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if japanese_chars > english_chars:
            return 'japanese'
        else:
            return 'english'
    
    def tokenize_text(self, text, language='auto'):
        """テキストをトークン化"""
        if language == 'auto':
            language = self.detect_language(text)
        
        if language == 'japanese':
            # 日本語の場合は数字と記号のみ除去（日本語文字は保持）
            text = re.sub(r'[0-9０-９]', '', text)
            text = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\s]', '', text)
            # 日本語の場合はjiebaを使用
            tokens = jieba.cut(text)
            tokens = [token for token in tokens if len(token) > 1 and token not in self.stop_words and token.strip()]
        else:
            # 英語の場合は従来通り
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # 英語の場合はNLTKを使用
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        return tokens
    
    def extract_cooccurrence(self, texts, window_size=5):
        """共起関係を抽出"""
        cooccurrence = defaultdict(int)
        
        for text in texts:
            tokens = self.tokenize_text(text)
            
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + window_size + 1, len(tokens))):
                    if tokens[i] != tokens[j]:
                        pair = tuple(sorted([tokens[i], tokens[j]]))
                        cooccurrence[pair] += 1
        
        return cooccurrence
    
    def get_top_words(self, texts, top_n=50):
        """頻出単語を取得"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text(text)
            all_tokens.extend(tokens)
        
        word_freq = Counter(all_tokens)
        return dict(word_freq.most_common(top_n))

def create_wordcloud(word_freq, title="ワードクラウド"):
    """ワードクラウドを作成"""
    if not word_freq:
        return None
    
    # 配置された日本語フォントファイルを使用
    font_path = "NotoSansJP-VariableFont_wght.ttf"
    
    try:
        # 日本語フォントを使用してワードクラウドを生成
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            font_path=font_path,
            max_words=100,
            max_font_size=100,
            min_font_size=10,
            prefer_horizontal=0.7,
            relative_scaling=0.5,
            collocations=False
        ).generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
        
    except Exception as e:
        # フォントファイルが見つからない場合のフォールバック
        try:
            st.warning("日本語フォントファイルが見つからないため、システムフォントを使用します。")
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=50,
                max_font_size=80,
                min_font_size=8,
                prefer_horizontal=0.8,
                relative_scaling=0.3,
                collocations=False,
                font_path=None
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"{title} (システムフォント版)", fontsize=16, fontweight='bold')
            
            return fig
        except Exception as e2:
            st.error(f"ワードクラウドの生成中にエラーが発生しました: {str(e2)}")
            return None

def create_cooccurrence_network(cooccurrence, min_weight=2, max_nodes=30):
    """共起ネットワークを作成"""
    try:
        if not cooccurrence:
            return None
        
        # 重みでフィルタリング
        filtered_cooccurrence = {k: v for k, v in cooccurrence.items() if v >= min_weight}
        
        if not filtered_cooccurrence:
            return None
    
        # ノードとエッジを準備
        G = nx.Graph()
        
        # エッジを追加
        for (word1, word2), weight in filtered_cooccurrence.items():
            G.add_edge(word1, word2, weight=weight)
        
        # ノード数が多すぎる場合は上位のものだけを残す
        if len(G.nodes()) > max_nodes:
            node_weights = defaultdict(int)
            for (word1, word2), weight in filtered_cooccurrence.items():
                node_weights[word1] += weight
                node_weights[word2] += weight
            
            top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = [node for node, _ in top_nodes]
            
            # 上位ノードに関連するエッジのみを残す
            edges_to_keep = [(word1, word2) for (word1, word2) in G.edges() 
                             if word1 in top_node_names and word2 in top_node_names]
            G = nx.Graph()
            for word1, word2 in edges_to_keep:
                weight = filtered_cooccurrence[(word1, word2) if word1 < word2 else (word2, word1)]
                G.add_edge(word1, word2, weight=weight)
        
        if len(G.nodes()) == 0:
            return None
        
        # レイアウトを計算
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Plotlyでネットワークを描画
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # ノードの詳細情報を作成
            degree = G.degree(node)
            neighbors = list(G.neighbors(node))
            neighbor_text = ', '.join(neighbors[:5])  # 最初の5つの隣接ノード
            if len(neighbors) > 5:
                neighbor_text += f'... (+{len(neighbors)-5}個)'
            
            hover_text = f"""
            <b>単語:</b> {node}<br>
            <b>次数:</b> {degree}<br>
            <b>隣接ノード:</b> {neighbor_text}
            """
            node_text.append(hover_text)
            
            # ノードサイズは次数に基づく
            node_size.append(max(10, degree * 5))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=[],
                line_width=2
            ),
            # ノードのドラッグ機能を有効化
            customdata=[node for node in G.nodes()],
            hovertemplate='%{text}<extra></extra>'
        )
        
        # ノードの色を次数に基づいて設定
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        node_trace.marker.color = node_adjacencies
    
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=dict(
                text='共起ネットワーク（インタラクティブ）',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            # インタラクティブ機能を追加
            dragmode='pan',  # ドラッグで移動
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)',
                color='black',
                activecolor='red'
            ),
            # ノードのドラッグ機能を強化
            clickmode='event+select',
            selectdirection='any'
        )
        
        # インタラクティブ機能を強化
        fig.update_traces(
            selector=dict(type='scatter'),
            hoverinfo='text+name',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial'
            ),
            # ノードの選択とドラッグ機能
            selected=dict(
                marker=dict(color='red', size=15)
            ),
            unselected=dict(
                marker=dict(opacity=0.8)
            )
        )
        
        # インタラクティブ機能を強化
        fig.update_layout(
            # ドラッグ機能を有効化
            dragmode='pan',
            # 選択機能を有効化
            selectdirection='any',
            # クリックイベントを有効化
            clickmode='event+select',
            # ノードの編集機能を有効化
            uirevision='constant'
        )
        
        return fig
    except Exception as e:
        st.error(f"共起ネットワークの作成中にエラーが発生しました: {str(e)}")
        return None



def main():
    st.markdown('<h1 class="main-header">📊 テキスト分析アプリ</h1>', unsafe_allow_html=True)
    
    # サイドバー
    st.sidebar.markdown("## 📁 データアップロード")
    uploaded_file = st.sidebar.file_uploader(
        "CSVファイルをアップロードしてください",
        type=['csv'],
        help="UTF-8エンコーディングのCSVファイルを推奨します"
    )
    
    # メインコンテンツ
    if uploaded_file is not None:
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(uploaded_file)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**ファイル名:** {uploaded_file.name}")
            st.markdown(f"**データサイズ:** {df.shape[0]}行 × {df.shape[1]}列")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # カラム選択
            st.markdown('<h2 class="sub-header">📋 分析対象カラムの選択</h2>', unsafe_allow_html=True)
            
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(text_columns) == 0:
                st.error("テキストデータを含むカラムが見つかりません。")
                return
            
            selected_columns = st.multiselect(
                "分析対象のカラムを選択してください（最大3つ）",
                text_columns,
                max_selections=3,
                default=text_columns[:min(3, len(text_columns))]
            )
            
            if not selected_columns:
                st.warning("分析対象のカラムを選択してください。")
                return
            
            # 分析設定
            st.markdown('<h2 class="sub-header">⚙️ 分析設定</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                language = st.selectbox(
                    "言語設定",
                    ["自動検出", "日本語", "英語"],
                    help="テキストの言語を指定してください"
                )
            
            with col2:
                min_word_length = st.slider(
                    "最小単語長",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="この長さ未満の単語は除外されます"
                )
            
            with col3:
                top_n_words = st.slider(
                    "表示する単語数",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="頻出単語の上位何個を表示するか"
                )
            
            # テキスト分析の実行
            analyzer = TextAnalyzer()
            
            # 選択されたカラムのテキストを結合
            all_texts = []
            for col in selected_columns:
                texts = df[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
            
            if not all_texts:
                st.error("分析可能なテキストデータが見つかりません。")
                return
            
            # 検索機能
            st.markdown('<h2 class="sub-header">🔍 検索機能</h2>', unsafe_allow_html=True)
            
            search_term = st.text_input(
                "キーワードを入力して検索",
                placeholder="例: 分析, データ, テキスト",
                help="入力したキーワードを含むテキストを検索します"
            )
            
            if search_term:
                matching_texts = []
                for text in all_texts:
                    if search_term.lower() in text.lower():
                        matching_texts.append(text)
                
                if matching_texts:
                    st.success(f"'{search_term}'を含むテキストを{len(matching_texts)}件見つけました。")
                    
                    # 検索結果の表示
                    with st.expander("検索結果を表示"):
                        for i, text in enumerate(matching_texts[:10]):  # 最初の10件のみ表示
                            st.write(f"{i+1}. {text[:200]}{'...' if len(text) > 200 else ''}")
                        
                        if len(matching_texts) > 10:
                            st.info(f"他に{len(matching_texts) - 10}件の結果があります。")
                else:
                    st.warning(f"'{search_term}'を含むテキストが見つかりませんでした。")
            
            # 分析実行ボタン
            if st.button("🚀 分析を実行", type="primary"):
                with st.spinner("テキストを分析中..."):
                    # 言語設定の適用
                    lang_setting = 'auto'
                    if language == "日本語":
                        lang_setting = 'japanese'
                    elif language == "英語":
                        lang_setting = 'english'
                    
                    # 単語頻度の分析
                    word_freq = analyzer.get_top_words(all_texts, top_n_words)
                    
                    # 共起関係の分析
                    cooccurrence = analyzer.extract_cooccurrence(all_texts)
                    
                    # 結果の表示
                    st.markdown('<h2 class="sub-header">📈 分析結果</h2>', unsafe_allow_html=True)
                    
                    # 基本統計
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("総テキスト数", len(all_texts))
                    
                    with col2:
                        st.metric("総単語数", sum(word_freq.values()))
                    
                    with col3:
                        st.metric("ユニーク単語数", len(word_freq))
                    
                    with col4:
                        if word_freq:
                            st.metric("平均単語長", f"{sum(len(word) * freq for word, freq in word_freq.items()) / sum(word_freq.values()):.1f}")
                    
                    # ワードクラウド
                    st.markdown('<h3 class="sub-header">☁️ ワードクラウド</h3>', unsafe_allow_html=True)
                    
                    if word_freq:
                        wordcloud_fig = create_wordcloud(word_freq)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                            plt.close()
                    else:
                        st.warning("ワードクラウドを生成するためのデータが不足しています。")
                    
                    # 頻出単語ランキング
                    st.markdown('<h3 class="sub-header">📊 頻出単語ランキング</h3>', unsafe_allow_html=True)
                    
                    if word_freq:
                        word_df = pd.DataFrame(list(word_freq.items()), columns=['単語', '出現回数'])
                        word_df = word_df.sort_values('出現回数', ascending=False)
                        
                        fig = px.bar(
                            word_df.head(20),
                            x='出現回数',
                            y='単語',
                            orientation='h',
                            title='上位20単語の出現回数'
                        )
                        fig.update_layout(
                            height=600,
                            title=dict(
                                text='上位20単語の出現回数',
                                font=dict(size=16)
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 共起ネットワーク
                    st.markdown('<h3 class="sub-header">🕸️ 共起ネットワーク</h3>', unsafe_allow_html=True)
                    
                    if cooccurrence:
                        network_fig = create_cooccurrence_network(cooccurrence)
                        if network_fig:
                            st.plotly_chart(network_fig, use_container_width=True)
                            
                            # インタラクティブ機能の説明
                            st.markdown('<h4 class="sub-header">🖱️ インタラクティブ機能</h4>', unsafe_allow_html=True)
                            st.info("""
                            **操作方法:**
                            - 🖱️ **ドラッグ**: ネットワーク全体を移動
                            - 🔍 **ズーム**: マウスホイールで拡大・縮小
                            - 📱 **タッチ**: スマートフォンでも操作可能
                            - ℹ️ **ホバー**: ノードにマウスを重ねると詳細情報を表示
                            - 🎯 **ノード選択**: ノードをクリックして選択
                            - ✋ **ノード移動**: 選択したノードをドラッグして移動
                            """)
                    else:
                        st.warning("共起ネットワークを生成するためのデータが不足しています。")
                    
                    # 詳細な共起関係
                    if cooccurrence:
                        st.markdown('<h3 class="sub-header">🔗 詳細な共起関係</h3>', unsafe_allow_html=True)
                        
                        cooccurrence_list = [(word1, word2, weight) for (word1, word2), weight in cooccurrence.items()]
                        cooccurrence_list.sort(key=lambda x: x[2], reverse=True)
                        
                        cooccurrence_df = pd.DataFrame(
                            cooccurrence_list[:50],
                            columns=['単語1', '単語2', '共起回数']
                        )
                        
                        st.dataframe(cooccurrence_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {str(e)}")
    
    else:
        st.markdown("""
        <div class="info-box">
        <h3>📋 使用方法</h3>
        <ol>
            <li>左サイドバーからCSVファイルをアップロードしてください</li>
            <li>分析対象のカラムを選択してください（最大3つ）</li>
            <li>分析設定を調整してください</li>
            <li>「分析を実行」ボタンをクリックしてください</li>
        </ol>
        
        <h3>🎯 機能</h3>
        <ul>
            <li>📊 ワードクラウド生成</li>
            <li>🕸️ 共起ネットワーク可視化</li>
            <li>🎬 アニメーション機能</li>
            <li>🔍 キーワード検索</li>
            <li>📈 頻出単語ランキング</li>
        </ul>
        
        <h3>📝 対応形式</h3>
        <ul>
            <li>CSVファイル（UTF-8エンコーディング推奨）</li>
            <li>日本語・英語テキスト対応</li>
            <li>複数カラム選択可能</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 