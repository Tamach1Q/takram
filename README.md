あなたはGIS・PDFベクター解析・Mapbox実装に強いエンジニアとして作業してください。

目的:
添付PDF「四国遍路ひとり歩き同行二人（地図編）」に載っているお遍路マップの赤い遍路道ルートを、Mapboxで表示・編集できるGeoJSONデータに起こす。

最終成果物:
1. 赤い遍路道ルートのGeoJSON
2. Mapboxで読み込める形式のサンプル
3. 抽出ログ
4. ページごとの抽出結果プレビュー画像
5. 失敗・曖昧な箇所の一覧
6. 後で手修正できる中間データ

重要方針:
画像認識から始めないこと。
まずPDF内のベクターデータを解析すること。
PyMuPDFの page.get_drawings() と page.get_text("dict") を使い、線の色、線幅、dash、bbox、pathを取得する。
OCRは最後のfallbackに限定する。

PDF:
PDFは/Desktop/takram/四国遍路ひとり歩き同行二人（地図編）第14版第1刷.pdf

対象:
地図ページを対象にする。
表紙、目次、解説のみのページ、宿泊施設リストなどは除外する。
ただし「この地図の見方」「凡例」は、赤線・線種・記号の仕様を理解するために必ず確認する。

赤い遍路道ルート抽出:
まず以下の条件を初期値として試す。

- 赤色の基準: RGB = (237, 28, 36)
- 正規化RGBなら、おおよそ (0.93, 0.11, 0.14)
- 色距離は L1 <= 0.08 を初期値
- dashed line を優先して抽出
- 主な候補:
  - width ≈ 1.25 pt
  - width ≈ 2.0 pt
  - dashes が "[ .05 2.5 ] 0" または "[ .02 3 ] 0" に近いもの
- 実線の赤線は単独採用しない
- dashed route の近く5pt以内にあり、長さ15pt以上の実線だけ補助的に接続対象にする
- 小さな丸、寺院記号、番号、吹き出し枠、凡例内の線はルートと混同しない
- type == "s" を優先
- type == "fs" はマーカー・記号・凡例の可能性が高いので別分類する

やること:
Step 1: PDF構造調査
- 各ページの get_drawings() を走査
- stroke color / fill color / width / dashes / bbox / path length を集計
- 赤系オブジェクトだけを一覧化
- ページごとに赤系線の総数、総延長、dash pattern、線幅の分布をCSVに出す
- 代表ページで抽出結果を画像に重ねてプレビューする

Step 2: 地図フレームの分割
- 1ページ内に複数地図がある場合、ページ全体を1枚の地図として扱わない
- 地図枠、グリッド、余白、凡例、解説欄を分離する
- frame_id を振る
- ルート線は frame_id ごとに管理する

Step 3: 赤線ルートのベクター化
- PDF pathをLineStringに変換
- Bezier curve がある場合は適切にdensifyする
- 細切れの破線を接続する
- 近接、角度、端点距離を使って同一ルート断片をマージする
- shapely / geopandas を使って中間GeoPackageまたはGeoJSONに保存する

Step 4: GCP候補の自動抽出
- 寺院名、札所番号、地名ラベルをPDFテキストから抽出する
- OCRではなく page.get_text("dict") を優先
- 赤文字ラベル、寺院名、札所番号、吹き出しを抽出する
- 寺院名のbbox中心ではなく、近くの寺院マーカー中心へsnapする
- 四国八十八箇所と別格霊場のgazetteerを作る
- gazetteerには寺院名、番号、緯度経度、別名、読みを含める
- ラベルとgazetteerを照合してGCP候補を作る
- GCP候補には confidence を持たせる

Step 5: PDF座標から地理座標への変換
- いきなりTPSを使わない
- GCP数に応じてモデルを選ぶ
  - GCP 3点: affineのみ
  - GCP 4〜5点: affine / projective を比較
  - GCP 6〜9点: affine / projective / polynomial2 を比較
  - GCP 10点以上かつ広く分布: TPSも比較対象に入れる
- JGD2011/JGD2024、平面直角座標系、WGS84系の候補を比較する
- 候補CRSと変換モデルを総当たりし、RMSE、LOOCV、seam errorで評価する
- GCPが少ないページは前後ページとの接続拘束を使う

Step 6: ページ間接続
- ページごとに独立して確定しない
- 前後ページのルート端点、共通寺院、共通地名、重複区間をtie pointとして使う
- seam error を計算する
- ページ間でルートが飛んでいる箇所を一覧化する
- Huber loss などを使い、外れGCPの影響を抑える

Step 7: Mapbox向け整形
- 最終GeoJSONは WGS84 / OGC:CRS84
- 座標順は [longitude, latitude]
- Feature properties には最低限以下を入れる

route_id
source_pdf
page_no
frame_id
segment_seq
style_class
transform_model
crs_candidate
gcp_count
rmse_m
loocv_rmse_m
seam_err_m
map_matched
confidence
needs_manual_review

style_class の例:
- walk_main
- walk_sub
- old_route
- car_route
- unknown_red
- legend_or_symbol

Step 8: Mapbox表示確認
- Mapbox GL JS で読み込むサンプルHTMLを作る
- ルートを赤線で表示する
- confidence が低い区間は破線または別レイヤーにする
- needs_manual_review=true の区間を目立たせる
- 元PDFページ画像を背景として重ねられる検証モードも用意する

禁止事項:
- いきなりラスター画像化して赤線だけをOpenCVで抜くこと
- OCRを最初から使うこと
- GCP 3〜5点でTPSを使うこと
- 凡例の赤線をルートとして混ぜること
- 寺院記号や赤文字をルート線として混ぜること
- ページ単位で雑に1つの変換を当てること
- RMSEだけで成功判定すること

検証基準:
- 抽出した赤線がPDF上の赤い遍路道に重なっていること
- 凡例や寺院記号が混入していないこと
- ページをまたぐルートが不自然に飛ばないこと
- Mapbox上で四国の実際の道路・歩道・地形と大きくズレないこと
- confidenceが低い箇所を隠さず、手修正対象として出力すること

まず最初にやってほしい作業:
1. PDFの全ページを走査して、赤系ベクターオブジェクトの統計を出す
2. 代表ページ3〜5枚で赤線抽出プレビュー画像を作る
3. ルート線、寺院記号、凡例、赤文字がどの条件で分離できそうか報告する
4. その後、抽出パイプラインのPythonスクリプトを作る

作業は一気に最終GeoJSONまで進めず、まず「赤線候補の抽出精度」を確認できるところまで実装してください。

## georef editor の兄弟リポジトリ参照

- `takram-image` 側の生成物は `takram` に丸ごとコピーせず、兄弟ディレクトリとして参照する。
- 想定構成:
  - `~/Desktop/takram`
  - `~/Desktop/takram-image`
- 今回のローカル例では `takram-image` 相当のリポジトリは `~/Desktop/henro` に置く。
- `takram/.env` に `TAKRAM_IMAGE_DIR=../henro` を設定する。
- georef editor は以下の入力を `TAKRAM_IMAGE_DIR/artifacts/panel_route_detection/` から読む。
  - `georef_panels.csv`
  - `georef_panels.geojson`
  - `accepted_main_georef_routes.geojson`
  - `accepted_inset_georef_routes.geojson`
- georef editor の選択単位は `layout_panel` ではなく `georef_panel`。
- `accepted_main_routes.geojson` のような旧 route 入力は使わず、`accepted_main_georef_routes.geojson` / `accepted_inset_georef_routes.geojson` を使う。
- 先に `takram-image` 側で `panel_route_detection` を実行してから、`takram` 側の georef editor を生成・利用する。

## manual route editor

- 自動抽出 route を最終採用せず、左の PDF crop を見ながら右の Mapbox 上で人間が直接 LineString を描く専用エディタを用意する。
- 入力は `TAKRAM_IMAGE_DIR/artifacts/panel_route_detection/` から読む。
  - `georef_panels.csv`
  - `georef_panels.geojson`
  - `accepted_main_georef_routes.geojson`
  - `accepted_inset_georef_routes.geojson`
- `accepted_*_georef_routes.geojson` は参考レイヤー扱いで、最終データにはしない。

### ローカル構成

- 想定構成:
  - `~/Desktop/takram`
  - `~/Desktop/henro`
- `takram/.env` に以下を設定する。
  - `TAKRAM_IMAGE_DIR=../henro`
  - `MAPBOX_ACCESS_TOKEN=...`

### build

- まず `henro` 側で `panel_route_detection` を実行しておく。
- その後 `takram` 側で manual route editor を生成する。

```bash
.venv/bin/python scripts/build_manual_route_editor.py \
  --pdf '四国遍路ひとり歩き同行二人（地図編）第14版第1刷.pdf' \
  --out-dir artifacts/manual_route_editor \
  --env-file .env
```

- 1つの georef panel だけを対象にした MVP 確認は `--georef-panel-id` を使う。

```bash
.venv/bin/python scripts/build_manual_route_editor.py \
  --pdf '四国遍路ひとり歩き同行二人（地図編）第14版第1刷.pdf' \
  --georef-panel-id G027_01 \
  --out-dir artifacts/manual_route_editor \
  --env-file .env
```

### 使い方

- `python -m http.server 8131 -d artifacts/manual_route_editor` で配信する。
- 最初に `Bind Output Dir` を押して `artifacts/manual_route_editor` を選ぶ。
- 左ペインは georef panel の PDF crop 参照用。
- 右ペインは Mapbox 手打ち用。PDF は左の参考で、最終ルートは右の Mapbox 上で描く。
- `Temple ON` の寺マーカーを目印にする。
- `Reference` route は壊れている可能性があるため、最終採用しない。
- `Save` で `manual_routes.geojson` に保存する。
- 右ペインの基本操作:
  - `click`: 点追加
  - `Shift + click`: 新しい LineString 開始
  - `double click`: 現在の LineString 確定
  - `Z`: Undo
  - `Backspace`: 最後の点削除
  - `S`: 保存
  - `N / P`: 次 / 前の panel
- 保存先をブラウザから直接更新したい場合は `Bind Output Dir` で `artifacts/manual_route_editor` を選ぶ。
- 保存される主な出力:
  - `artifacts/manual_route_editor/manual_routes.geojson`
  - `artifacts/manual_route_editor/autosave.json`
  - `artifacts/manual_route_editor/route_segments/`
